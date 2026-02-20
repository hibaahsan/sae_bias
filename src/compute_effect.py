import argparse
import pickle
import torch as t
import pandas as pd
import numpy as np
from nnsight import LanguageModel
from loading_utils import Submodule, get_prompts, get_saes_and_submodules, load_data, logit_diff_metric_fn, get_mask
from transformers import AutoTokenizer
import hashlib
from attribution import patching_effect
import gc
import os


def main():
    parser = argparse.ArgumentParser(description="Get logit diff")
    parser.add_argument('-is_9b', type=int, help='Gemma 9B?')
    parser.add_argument('-task', type=str, help='Task name')
    args = parser.parse_args()
    

    task = args.task.replace('-', ' ')
    is_9b = bool(args.is_9b)
    model_size = '9B' if is_9b else '2B'
    device = 'cuda:0'

    print(task)

    if is_9b:
        model_name = 'google/gemma-2-9b-it'
        layers=[20]

    else:
        model_name = 'google/gemma-2-2b-it'
        layers=[12]


    notes_df, text_col = load_data(task)


    model = LanguageModel(model_name, torch_dtype=t.bfloat16, load_in_4bit=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    no_token_ix = t.tensor(tokenizer('No', add_special_tokens=False)['input_ids'], device=device)
    yes_token_ix = t.tensor(tokenizer('Yes', add_special_tokens=False)['input_ids'], device=device)
    print(no_token_ix, yes_token_ix)

    dictionaries, submodules = get_saes_and_submodules(model, layers, is_9b=is_9b)

    if task == 'qpain':
        pred=yes_token_ix
        target=no_token_ix
    else:
        pred=no_token_ix
        target=yes_token_ix

    all_note_id = []
    all_feat = []
    all_layer = []
    all_act = []

    digests = []

    steps=3

    os.makedirs("../tmp", exist_ok=True)

    print('Getting latents')

    for ix, row in notes_df.iterrows():
        text = row[text_col]
        chat_text, _, _ = get_prompts(text, task, tokenizer, False)

        hash_input = [text, task] + [s.name for s in submodules]
        hash_str = ''.join(hash_input)
        hash_digest = hashlib.md5(hash_str.encode()).hexdigest()

        effects, *_ = patching_effect(
            [chat_text],
            None,
            model,
            submodules,
            dictionaries,
            logit_diff_metric_fn,
            metric_kwargs=dict(pred=pred, target=target),
            method='ig',
            steps=steps)

        to_save = {
                k.name : v.detach().to("cpu") for k, v in effects.items()
                }

        t.save(to_save, f"../tmp/{hash_digest}.pt")
        digests.append(hash_digest)

        del effects, _
        gc.collect()


    aggregated_effects = {submodule.name : 0 for submodule in submodules}

    for hash_digest in digests:
        effects = t.load(f"../tmp/{hash_digest}.pt", weights_only=False)
        for submodule in submodules:
            n = effects[submodule.name].act.shape[1]
            aggregated_effects[submodule.name] += (
                        effects[submodule.name].act[:,4:-5,:] # remove BOS features
                    ).sum(dim=1).mean(dim=0)


    size = len(digests)
    aggregated_effects = {k : v / size for k, v in aggregated_effects.items()}


    count = 0
    for k, v in aggregated_effects.items():
        kk = int(k.split('_')[-1])

        for idx in (v > 0.0).nonzero():
            count += 1
            all_feat.append(idx.item())
            all_layer.append(kk)
            all_act.append(v[idx].item())

    feat_df = pd.DataFrame.from_dict({
        'layer': all_layer,
        'latent': all_feat,
        'effect': all_act,
        })

    feat_df = feat_df.sort_values(by=['effect'], ascending=False)
    feat_df.to_csv(f'../results/latents_{task}_{model_size}.csv', sep='\t', index=False)

    

if __name__ == '__main__':
    main()