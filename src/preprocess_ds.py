import re
import argparse
import pandas as pd
from transformers import AutoTokenizer
from loading_utils import get_token_count


def preprocess_text(row_text):
    row_text = re.sub(r'\[.*?\]', '___', row_text)
    return re.sub(r'\s+', ' ', row_text).strip()

def load_admissions(mimic_path):
    df_adm = pd.read_csv(f'{mimic_path}/ADMISSIONS.csv.gz')
    for col in ('ADMITTIME', 'DISCHTIME', 'DEATHTIME'):
        df_adm[col] = pd.to_datetime(df_adm[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    df_adm = df_adm[
        (df_adm['ADMISSION_TYPE'] != 'NEWBORN') &
        df_adm['DEATHTIME'].isnull()
    ]
    return df_adm.sort_values(['SUBJECT_ID', 'ADMITTIME']).reset_index(drop=True)


def load_discharge_notes(mimic_path):
    df_notes = pd.read_csv(f'{mimic_path}/NOTEEVENTS.csv.gz', low_memory=False)
    df_notes = df_notes.sort_values(['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])
    return df_notes[
        (df_notes['CATEGORY'] == 'Discharge summary') &
        (df_notes['DESCRIPTION'] == 'Report')
    ]


def build_notes_df(mimic_path):
    """Merge admissions, discharge notes, and patient DOB; compute age; filter adults."""
    df_adm = load_admissions(mimic_path)
    df_discharge = load_discharge_notes(mimic_path)

    df = pd.merge(
        df_adm[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ETHNICITY']],
        df_discharge[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'TEXT', 'CATEGORY', 'DESCRIPTION']],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='left',
    )

    # Keep only the last discharge summary per admission (handles duplicates)
    df = df.groupby(['SUBJECT_ID', 'HADM_ID']).nth(-1).reset_index()
    df = df[df['TEXT'].notnull()]

    patients_df = pd.read_csv(f'{mimic_path}/PATIENTS.csv')
    patients_df['DOB'] = pd.to_datetime(patients_df['DOB'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    df = df.merge(patients_df[['SUBJECT_ID', 'DOB']], on='SUBJECT_ID', how='inner')
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME']).dt.date
    df['DOB'] = pd.to_datetime(df['DOB']).dt.date
    df['AGE'] = df.apply(lambda row: (row['ADMITTIME'] - row['DOB']).days // 365, axis=1)

    return df[df['AGE'] > 18].copy()


def split_by_race(df_adm_notes):
    """Filter to Black/White patients and create balanced train/val/test splits."""
    df = df_adm_notes[
        df_adm_notes['ETHNICITY'].isin(['WHITE', 'BLACK/AFRICAN AMERICAN'])
    ].copy()
    df['IS_BLACK'] = (df['ETHNICITY'] == 'BLACK/AFRICAN AMERICAN').astype(int)

    black_ids = df[df['IS_BLACK'] == 1]['SUBJECT_ID']
    white_ids = df[df['IS_BLACK'] == 0]['SUBJECT_ID'].sample(n=len(black_ids), random_state=1)

    def make_split(ids_pos, ids_neg, val_frac=0.1, test_frac=0.1):
        val_t = ids_pos.sample(frac=val_frac + test_frac, random_state=1)
        train_t = ids_pos.drop(val_t.index)
        test_t = val_t.sample(frac=0.5, random_state=1)
        val_t = val_t.drop(test_t.index)

        val_f = ids_neg.sample(frac=val_frac + test_frac, random_state=1)
        train_f = ids_neg.drop(val_f.index)
        test_f = val_f.sample(frac=0.5, random_state=1)
        val_f = val_f.drop(test_f.index)

        train_ids = pd.concat([train_t, train_f])
        val_ids = pd.concat([val_t, val_f])
        test_ids = pd.concat([test_t, test_f])
        return train_ids, val_ids, test_ids

    train_ids, val_ids, test_ids = make_split(black_ids, white_ids)

    bw_train = df[df['SUBJECT_ID'].isin(train_ids)].copy()
    bw_val = df[df['SUBJECT_ID'].isin(val_ids)].copy()
    bw_test = df[df['SUBJECT_ID'].isin(test_ids)].copy()

    return bw_train, bw_val, bw_test


def add_preprocessed_cols(df, tokenizer):
    df['PREPROCESS_TEXT'] = df['TEXT'].apply(preprocess_text)
    df['TOK_COUNT'] = df['PREPROCESS_TEXT'].apply(lambda x: get_token_count(x, tokenizer))
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC discharge summaries")
    parser.add_argument('-mimic_path', type=str, help='Path to MIMIC III repository')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')

    print('Loading and merging data...')
    df_adm_notes = build_notes_df(args.mimic_path)

    print('Splitting by race...')
    bw_train, bw_val, bw_test = split_by_race(df_adm_notes)

    print('Preprocessing text and counting tokens...')
    bw_train = add_preprocessed_cols(bw_train, tokenizer)
    bw_val = add_preprocessed_cols(bw_val, tokenizer)
    bw_test = add_preprocessed_cols(bw_test, tokenizer)

    bw_train.to_csv('../data/discharge_bw_train.csv', sep='\t', index=False)
    bw_val.to_csv('../data/discharge_bw_val.csv', sep='\t', index=False)
    bw_test.to_csv('../data/discharge_bw_test.csv', sep='\t', index=False)
    print('Done.')


if __name__ == '__main__':
    main()
