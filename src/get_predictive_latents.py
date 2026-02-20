import argparse
import gc

import pandas as pd
import numpy as np
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from nnsight import LanguageModel
from loading_utils import get_saes_and_submodules


def get_coeff(train_latents, train_labels, test_latents, test_labels, C, top_k=100, max_iter=1000):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_latents)
    X_test = scaler.transform(test_latents)

    clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=max_iter, C=C)
    clf.fit(X_train, train_labels)
    coeff = clf.coef_.flatten()

    sorted_ixs = np.argsort(-np.abs(coeff)).tolist()
    sorted_coeffs = coeff[sorted_ixs].tolist()

    y_prob = clf.predict_proba(X_test)
    auroc = roc_auc_score(test_labels, y_prob[:, 1])
    print(f"AUROC: {auroc:.4f}")

    return sorted_ixs[:top_k], sorted_coeffs[:top_k]


def get_max_agg_latents(texts, model, submodules, dictionaries, gc_every=100):
    """Extract per-text max-pooled SAE latent activations.
    """
    all_max_latents = []

    with t.no_grad():
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f'batch {i}')

            with model.trace([text]):
                for submodule in submodules:
                    dictionary = dictionaries[submodule]
                    x = submodule.get_activation()
                    _, f = dictionary(x, output_features=True)
                    # max-pool over token dim (skip BOS at index 0), save scalar per feature
                    max_latent = f[:, 1:, :].max(dim=1)[0].save()

            all_max_latents.append(max_latent.squeeze())

            if i % gc_every == 0:
                gc.collect()
                t.cuda.empty_cache()

    return t.stack(all_max_latents).cpu().float()


def main():
    parser = argparse.ArgumentParser(description="Get predictive latents")
    parser.add_argument('-is_9b', type=int, help='Gemma 9B?')
    parser.add_argument('-layer', type=int, help='layer')
    args = parser.parse_args()

    is_9b = bool(args.is_9b)
    layer = args.layer
    model_size = '9B' if is_9b else '2B'
    print(layer, is_9b)

    model_name = 'google/gemma-2-9b-it' if is_9b else 'google/gemma-2-2b-it'
    model = LanguageModel(model_name, torch_dtype=t.bfloat16, load_in_4bit=True, device_map='auto')

    dictionaries, submodules = get_saes_and_submodules(model, [layer], is_9b=is_9b)

    TEXT_COL = 'PREPROCESS_TEXT'
    TASK = 'discharge_bw'
    GT_COL = 'IS_BLACK'

    train_df = pd.read_csv(f"../data/{TASK}_train.csv", sep='\t')
    test_df = pd.read_csv(f"../data/{TASK}_test.csv", sep='\t')

    train_df = train_df[train_df['TOK_COUNT'] < 2000].groupby([GT_COL]).head(500).copy()
    test_df = test_df[test_df['TOK_COUNT'] < 2000].groupby([GT_COL]).head(100).copy()

    train_texts = list(train_df[TEXT_COL])
    train_labels = np.array(train_df[GT_COL])
    test_texts = list(test_df[TEXT_COL])
    test_labels = np.array(test_df[GT_COL])

    print('Computing train latents...')
    train_latents = get_max_agg_latents(train_texts, model, submodules, dictionaries)

    print('Computing test latents...')
    test_latents = get_max_agg_latents(test_texts, model, submodules, dictionaries)

    ixs, coeffs = get_coeff(train_latents, train_labels, test_latents, test_labels, C=1)

    latent_df = pd.DataFrame({'latent': ixs, 'coeff': coeffs})
    latent_df.to_csv('../results/top_pred_latents_{model_size}.csv',
        sep='\t', index=False)


if __name__ == '__main__':
    main()
