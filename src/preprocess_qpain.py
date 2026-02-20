import pandas as pd


# Taken from Q_Pain_Experiments.ipynb

NAMES = {
    "Black": {
        "man":   ["Roosevelt", "Jermaine", "Darnell", "Willie", "Mattie",
                  "Reginald", "Cedric", "Sylvester", "Tyrone", "Errol"],
        "woman": ["Latonya", "Tamika", "Ebony", "Latasha", "Keisha",
                  "Lillie", "Minnie", "Gwendolyn", "Bessie", "Marva"],
    },
    "White": {
        "man":   ["Bradley", "Brett", "Scott", "Kurt", "Todd", "Chad",
                  "Matthew", "Dustin", "Shane", "Douglas"],
        "woman": ["Beth", "Megan", "Kristin", "Jill", "Erin", "Colleen",
                  "Kathleen", "Heather", "Holly", "Laurie"],
    },
}

PRONOUNS = {
    "subject":    {"man": "he",  "woman": "she"},
    "possessive": {"man": "his", "woman": "her"},
}

RACES = ["Black", "White"]
GENDERS = ["man", "woman"]

DATA_FILES = [
    '../data/q-pain/1.0.0/data_acute_cancer.csv',
    '../data/q-pain/1.0.0/data_acute_non_cancer.csv',
    '../data/q-pain/1.0.0/data_chronic_cancer.csv',
    '../data/q-pain/1.0.0/data_chronic_non_cancer.csv',
    '../data/q-pain/1.0.0/data_post_op.csv',
]


def create_open_standard(data, q):
    return (
        "Vignette: " + data.Vignette[q] + "\n" +
        "Question: " + data.Question[q] + "\n" +
        "Answer:"
    )


def genderize_open(prompt, gender):
    return (prompt
            .replace('[gender]', gender)
            .replace('[possessive]', PRONOUNS['possessive'][gender])
            .replace('[subject]', PRONOUNS['subject'][gender]))


def race_name_open(prompt, race, gender, q):
    return (prompt
            .replace('[race]', race)
            .replace('Patient D', NAMES[race][gender][q]))


def build_vignette_pair(prompt_standard, gender, q):
    """Return (black_vignette, white_vignette) for a given gender and question index."""
    prompt_gendered = genderize_open(prompt_standard, gender)
    vignettes = {}
    for race in RACES:
        p = race_name_open(prompt_gendered, race, gender, q)
        p = p.replace('Vignette: ', '')
        p = p.split('?')[0] + "? Respond with 'Yes' or 'No' only."
        vignettes[race] = p
    return vignettes['Black'], vignettes['White']


def main():
    black_arr, caucasian_arr, gender_arr, filename_arr = [], [], [], []

    for filename in DATA_FILES:
        vignettes = pd.read_csv(filename)
        data = vignettes[vignettes.Answer.str.contains("Yes")].reset_index()
        print(filename, len(data))

        for q in range(10):
            prompt_standard = create_open_standard(data, q)
            for gender in GENDERS:
                black_vignette, white_vignette = build_vignette_pair(prompt_standard, gender, q)
                black_arr.append(black_vignette)
                caucasian_arr.append(white_vignette)
                gender_arr.append(gender)
                filename_arr.append(filename)

    pain_df = pd.DataFrame({
        'aa_vignette': black_arr,
        'caucasian_vignette': caucasian_arr,
        'gender': gender_arr,
        'filename': filename_arr,
    })

    pain_df.to_csv('../data/qpain_processed.csv', sep='\t', index=False)


if __name__ == '__main__':
    main()
