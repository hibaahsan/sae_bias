import argparse
import pandas as pd


AA_TOKEN = "African-American"


def insert_token(row):
    pos = row["position"]
    return row["brief_hospital_course"][:pos] + AA_TOKEN + ' ' + row["brief_hospital_course"][pos:]


def select_rows(note_di_df, indices_df):
    filtered_df = note_di_df[['brief_hospital_course', 'note_id']].merge(indices_df, on='note_id')
    filtered_df['explicit_bhc'] = filtered_df.apply(insert_token, axis=1)
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess BHCs")
    parser.add_argument('-bhcs_path', type=str, help='Path to BHCs CSV')

    args = parser.parse_args()

    note_di_df = pd.read_csv(args.bhcs_path)

    datasets = [
        ('cocaine abuse', '../data/bhc_cocaine_abuse.csv', 'explicit_bhc_cocaine_abuse.csv'),
        ('pregnant patients', '../data/bhc_pregnant.csv', 'explicit_bhc_pregnant.csv'),
    ]

    for label, index_path, out_filename in datasets:
        indices_df = pd.read_csv(index_path, sep='\t')
        df = select_rows(note_di_df, indices_df)
        df.to_csv(f'../data/{out_filename}', sep='\t', index=False)
        print(f'# BHCs for {label}: {len(df)}')


if __name__ == '__main__':
    main()
