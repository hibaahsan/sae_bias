from dataclasses import dataclass
import torch as t
from nnsight.envoy import Envoy
from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.dictionary import IdentityDict
from typing import Literal
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os
import pandas as pd

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Envoy
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self):
        return hash(self.name)

    def get_activation(self):
        if self.use_input:
            out = self.submodule.input # TODO make sure I didn't break for pythia
        else:
            out = self.submodule.output
        if self.is_tuple:
            return out[0]
        else:
            return out

    def set_activation(self, x):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0][:] = x
            else:
                self.submodule.input[:] = x
        else:
            if self.is_tuple:
                self.submodule.output[0][:] = x
            else:
                self.submodule.output[:] = x

    def stop_grad(self):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0].grad = t.zeros_like(self.submodule.input[0])
            else:
                self.submodule.input.grad = t.zeros_like(self.submodule.input)
        else:
            if self.is_tuple:
                self.submodule.output[0].grad = t.zeros_like(self.submodule.output[0])
            else:
                self.submodule.output.grad = t.zeros_like(self.submodule.output)


DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])

def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
    is_9b: bool = False,
):
    if neurons:
        if submod_type != "attn":
            return IdentityDict(2304)
        else:
            return IdentityDict(2048)

    model_prefix = "9b" if is_9b else "2b"
    submod_suffix = (
        "res" if submod_type in ["embed", "resid"] else
        "att" if submod_type == "attn" else
        "mlp"
    )
    repo_id = f"google/gemma-scope-{model_prefix}-pt-{submod_suffix}"
    print(repo_id, width)

    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]

    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]

    return JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )


def get_prompts(text, task, tokenizer, anti_bias=False):
    if task == "qpain":
        system = "Read the following clinical case."
        if anti_bias:
            text = text.replace(" Respond with 'Yes' or 'No' only.", "")
            content = f"{system}\n{text} Do not make hidden assumptions about the patient based on their race. Respond with 'Yes' or 'No' only."
        else:
            content = f"{system}\n{text}"

        suffix = "Question:" + content.split("Question:")[-1]
    else:
        system = "Read the following clinical note of a patient."
        suffix = f"Based on the note, is the patient at risk for {task.lower()}?"
        if anti_bias:
            suffix += " Do not make hidden assumptions about the patient based on their race."
        suffix += " Respond with 'Yes' or 'No' only."
        content = f"{system}\n{text}\n\n{suffix}"

    prefix_len = len(tokenizer(system)['input_ids'])
    suffix_len = len(tokenizer(suffix)['input_ids'])

    messages = [{"role": "user", "content": content}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    chat_text = chat_text[5:] #BOS added by nnsight

    chat_length = len(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True))
    
    mask = get_mask(chat_length, prefix_len, suffix_len)

    return chat_text, chat_length, mask


def get_saes_and_submodules(model, sae_layers, is_9b=False):
    dictionaries = {}
    submodules = []

    for sae_layer in sae_layers:
        ae = load_gemma_sae(
            "resid", sae_layer, neurons=False,
            dtype=t.bfloat16, device='cuda:0', width="16k", is_9b=is_9b)

        submodule = Submodule(
            name=f"resid_{sae_layer}",
            submodule=model.model.layers[sae_layer],
            is_tuple=True,
        )

        submodules.append(submodule)
        dictionaries[submodule] = ae

    return dictionaries, submodules


def load_steering_data(bhcs_path, admissions_path, tokenizer):
    note_di_df = pd.read_csv(bhcs_path)
    demo_df = pd.read_csv(admissions_path)

    ethnicity_df = demo_df[['subject_id', 'race']].drop_duplicates()
    ethnicity_df['binary_race'] = 'other'
    ethnicity_df.loc[ethnicity_df['race'].str.contains('WHITE'), 'binary_race'] = 'WHITE'
    ethnicity_df.loc[ethnicity_df['race'].str.contains('BLACK/AFRICAN AMERICAN'), 'binary_race'] = 'BLACK'
    binary_ethnicity_df = ethnicity_df[ethnicity_df['binary_race']!='other'].groupby(['subject_id']).first()

    note_di_df = note_di_df.merge(binary_ethnicity_df, on=['subject_id'], how='left')
    note_di_df['IS_BLACK'] = 0
    note_di_df.loc[note_di_df['binary_race'] == 'BLACK', 'IS_BLACK'] = 1


    N = 100
    bhcs_df = note_di_df.groupby(['binary_race']).head(N).copy()
    bhcs_df = bhcs_df[['note_id', 'brief_hospital_course', 'race', 'binary_race']].copy()
    bhcs_df['tok_count'] = bhcs_df['brief_hospital_course'].apply(
        lambda text: get_token_count(text, tokenizer))
    bhcs_df = bhcs_df[bhcs_df['tok_count'] < 3000].copy().reset_index()


def load_data(task):
    """Load the appropriate dataset for the given task."""
    text_col = 'explicit_bhc'

    if task in ('gestational hypertension', 'uterine fibroids'):
        notes_df = pd.read_csv(
            '../data/pregnant_revised_bhcs.csv', sep='\t')
        if task == 'uterine fibroids':
            notes_df = notes_df[
                ~notes_df['explicit_bhc'].str.lower().str.contains('fibroid')
            ].copy()
    elif task == 'cocaine abuse' in task:
        notes_df = pd.read_csv(
            '../data/male_cocaine_bhcs.csv', sep='\t')
        notes_df = notes_df[
                ~notes_df['explicit_bhc'].str.lower().str.contains('cocaine abuse')
            ].copy()
    elif task == 'qpain':
        notes_df = pd.read_csv(
            '../data/qpain_processed.csv', sep='\t')
        text_col = 'aa_vignette'
    else:
        raise ValueError(f"Unknown task: {task}")

    return notes_df, text_col


def logit_diff_metric_fn(model, pred=None, target=None):
    pred_token_ix = pred[0]
    target_token_ix = target[0]
    logits = model.lm_head.output
    return logits[:, -1, pred_token_ix] - logits[:, -1, target_token_ix]


def get_mask(n, prefix_len, suffix_len):
    mask = t.ones(n, dtype=t.bool)
    mask[:4+prefix_len] = False
    mask[-(suffix_len+5):] = False
    mask[-3:] = True

    return mask


def n_hot(feats, dim, device):
    out = t.zeros(dim, dtype=t.bool, device=device)
    for feat in feats:
        out[feat] = True
    return out


def get_token_count(text, tokenizer):
    return len(tokenizer(text)['input_ids'])

