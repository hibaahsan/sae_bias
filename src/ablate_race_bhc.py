import argparse
import pickle
import torch as t
import pandas as pd
import numpy as np
from nnsight import LanguageModel
from loading_utils import Submodule, get_prompts, get_saes_and_submodules, load_data, n_hot
from transformers import AutoTokenizer
from scipy import stats
import json



def print_bias_analysis(results):
    """Print formatted results of bias analysis."""
    print("=" * 60)
    print(f"BIAS ANALYSIS: {results['race1_name']} vs {results['race2_name']}")
    print("=" * 60)

    print(f"\nSample Size: {results['n_pairs']} matched pairs")

    print(f"\nLOGIT DIFFERENCES ({results['race1_name']} - {results['race2_name']}):")
    print(f"  Mean: {results['mean_logit_diff']:.4f}")
    print(f"  Std:  {results['std_logit_diff']:.4f}")
    print(f"  95% CI: [{results['ci_95'][0]:.4f}, {results['ci_95'][1]:.4f}]")

    print(f"\nSTATISTICAL TESTS:")
    print(f"  t-test:")
    print(f"    t = {results['ttest_statistic']:.4f}")
    print(f"    p = {results['ttest_pvalue']:.6f}")

    alpha = 0.05
    print(f"\nINTERPRETATION (α = {alpha}):")

    if results['ttest_pvalue'] < alpha:
        direction = results['race1_name'] if results['mean_logit_diff'] > 0 else results['race2_name']
        print(f"  ✓ SIGNIFICANT BIAS detected favoring {direction}")
    else:
        print(f"  ✗ No significant bias detected")

    if results['mean_logit_diff'] > 0:
        print(f"  Direction: Higher probability assigned to {results['race1_name']}")
    elif results['mean_logit_diff'] < 0:
        print(f"  Direction: Higher probability assigned to {results['race2_name']}")
    else:
        print(f"  Direction: No systematic difference")



def analyze_logit_differences(race1_logits, race2_logits,
                              race1_name="African-American",
                              race2_name="Caucasian"):
    """Analyze bias using logit differences between matched counterfactual pairs."""
    logit_differences = race1_logits - race2_logits

    t_stat, t_pvalue = stats.ttest_rel(race1_logits, race2_logits)

    mean_diff = np.mean(logit_differences)
    std_diff = np.std(logit_differences, ddof=1)

    n = len(logit_differences)
    se = std_diff / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, n - 1, mean_diff, se)

    return {
        'n_pairs': n,
        'logit_differences': logit_differences,
        'mean_logit_diff': mean_diff,
        'std_logit_diff': std_diff,
        'ci_95': ci_95,
        'ttest_statistic': t_stat,
        'ttest_pvalue': t_pvalue,
        'race1_name': race1_name,
        'race2_name': race2_name,
    }



def run_inference(model, tokenizer, notes_df, text_col, task, submodules,
                  dictionaries, feats_to_ablate, yes_token_ix, no_token_ix, device):
    """Run model inference for all intervention types and collect logits."""
    softmax = t.nn.Softmax(dim=-1)
    generate_kwargs = dict(do_sample=False)
    max_new_tokens = 1

    intervention_types = [
        'African-American', 'Caucasian',
        'African-American-anti-bias', 'Caucasian-anti-bias',
        'African-American-ablated',
    ]

    logits_dict = {}
    probs_dict = {}

    for col_name in intervention_types:
        logits_dict[col_name] = []
        probs_dict[col_name] = []

        for ix, row in notes_df.iterrows():
            text = row[text_col]

            if 'Caucasian' in col_name:
                if task == 'qpain':
                    text = row['caucasian_vignette']
                else:
                    text = text.replace('African-American', 'Caucasian')

            anti_bias = 'anti-bias' in col_name
            chat_text,_,_ = get_prompts(text, task, tokenizer, anti_bias)

            if ix == 0:
                print(chat_text)

            if col_name != 'African-American-ablated':
                with t.no_grad():
                    with model.generate(max_new_tokens=max_new_tokens, **generate_kwargs) as tracer:
                        with tracer.invoke(chat_text) as invoker:
                            logits = model.output.logits.squeeze()
                            probs = softmax(logits)
                            prob_diff = (probs[yes_token_ix][0] - probs[no_token_ix][0]).save()
                            logit_diff = (logits[yes_token_ix][0] - logits[no_token_ix][0]).save()
                            logits_dict[col_name].append(logit_diff)
                            probs_dict[col_name].append(prob_diff)
            else:
                with t.no_grad():
                    with model.generate(max_new_tokens=max_new_tokens, **generate_kwargs) as tracer:
                        with tracer.invoke(chat_text) as invoker:
                            for submodule in submodules:
                                dictionary = dictionaries[submodule]
                                feat_add_idxs = feats_to_ablate[submodule]

                                x = submodule.get_activation()
                                x_hat, f = dictionary(x, output_features=True)
                                res = x - x_hat
                                f[:, :, feat_add_idxs] = 0.
                                submodule.set_activation(dictionary.decode(f) + res)

                            logits = model.output.logits.squeeze()
                            probs = softmax(logits)
                            prob_diff = (probs[yes_token_ix][0] - probs[no_token_ix][0]).save()
                            logit_diff = (logits[yes_token_ix][0] - logits[no_token_ix][0]).save()
                            logits_dict[col_name].append(logit_diff)
                            probs_dict[col_name].append(prob_diff)

    # Move logits to CPU
    num_samples = len(notes_df)
    cpu_logits_dict = {}
    for col_name in intervention_types:
        cpu_logits_dict[col_name] = [
            logits_dict[col_name][i].cpu().float() for i in range(num_samples)
        ]

    return cpu_logits_dict, intervention_types


def run_analysis(cpu_logits_dict, task, results_path):
    """Run bias analysis on collected logits and save results."""
    # 1. Explicit race bias
    race1_logits = np.array(cpu_logits_dict['African-American'])
    race2_logits = np.array(cpu_logits_dict['Caucasian'])
    results = analyze_logit_differences(race1_logits, race2_logits, 'African-American', 'Caucasian')
    print_bias_analysis(results)
    results['task'] = task

    # 2. Anti-bias prompt
    race1_logits = np.array(cpu_logits_dict['African-American-anti-bias'])
    race2_logits = np.array(cpu_logits_dict['Caucasian-anti-bias'])
    anti_bias_results = analyze_logit_differences(
        race1_logits, race2_logits, 'African-American-anti-bias', 'Caucasian-anti-bias')
    print_bias_analysis(anti_bias_results)
    results['anti_bias_results'] = anti_bias_results

    # 3. Ablated vs Caucasian
    race1_logits = np.array(cpu_logits_dict['African-American-ablated'])
    race2_logits = np.array(cpu_logits_dict['Caucasian'])
    ablated_results = analyze_logit_differences(
        race1_logits, race2_logits, 'African-American-ablated', 'Caucasian')
    print_bias_analysis(ablated_results)
    results['ablated_results'] = ablated_results

    with open(results_path, 'w') as f:
        pickle.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description="Get logit diff")
    parser.add_argument('-is_9b', type=int, help='Gemma 9B?')
    parser.add_argument('-task', type=str, help='Task name')
    parser.add_argument('-multiple', type=int, help='ablate multiple layers')

    args = parser.parse_args()

    task = args.task.replace('-', ' ')
    is_9b = bool(args.is_9b)
    multiple = bool(args.multiple)
    model_size = '9B' if is_9b else '2B'
    suffix = 'multiple' if multiple else ''
    device = 'cuda:0'

    results_path = f'../results/results_{task}_{model_size}_{suffix}.p'

    if is_9b:
        model_name = 'google/gemma-2-9b-it'
        feats_to_ablate_dict = {
            'resid_20': [426, 2577, 7757, 10081, 13114, 13578, 14766, 14319, 15070],
        }

        if multiple:
            print('Intervening on multiple layers')
            feats_to_ablate_dict = {'resid_20' :[426,2577,7757,10081,13114,13578,14766,14319,15070,], 
                                'resid_21': [1655],
                                'resid_22': [5325,537,5990,8681],
                                'resid_23': [12193,13864,1665,5163],
                                'resid_24': [11343,2988,3457,6636]}


    else:
        model_name = 'google/gemma-2-2b-it'
        feats_to_ablate_dict = {
            'resid_12': [4185, 6364, 10263, 11573, 3718, 7137, 7192],
        }

        if multiple:
            print('Intervening on multiple layers')
            feats_to_ablate_dict = {'resid_12' :[4185,6364,10263,11573,3718,7137,7192], 
                                'resid_13': [11457, 5413, 5901, 6186, 7338],
                                'resid_14': [13380, 2136, 5483, 8433],
                                'resid_15': [13891, 1546, 16347, 8395, 9522],
                                'resid_16': [10624, 11234, 13818, 13893, 14349, 6295, 7823]}

    layers = [int(resid.split('_')[-1]) for resid in feats_to_ablate_dict]
    print(layers)

    notes_df, text_col = load_data(task)

    model = LanguageModel(model_name, torch_dtype=t.bfloat16, load_in_4bit=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    no_token_ix = t.tensor(tokenizer('No', add_special_tokens=False)['input_ids'], device=device)
    yes_token_ix = t.tensor(tokenizer('Yes', add_special_tokens=False)['input_ids'], device=device)
    print(no_token_ix, yes_token_ix)

    dictionaries, submodules = get_saes_and_submodules(model, layers, is_9b=is_9b)


    feats_to_ablate = {
        submodule: n_hot(feats_to_ablate_dict[submodule.name], dictionaries[submodule].dict_size, device)
        for submodule in submodules
    }

    results, _ = run_inference(
        model, tokenizer, notes_df, text_col, task,
        submodules, dictionaries, feats_to_ablate,
        yes_token_ix, no_token_ix, device)

    run_analysis(results, task, results_path)


if __name__ == '__main__':
    main()
