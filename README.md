# sae_bias
Code repository for ["Can SAEs reveal and mitigate racial biases of LLMs in healthcare?"](https://arxiv.org/pdf/2511.00177)

## Setting up environment

```conda env create -f environment.yml```

## Predictive latents
1. Run ```python src/preprocess_ds.py -mimic_path  <mimic_iii_path>``` to preprocess [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) discharge summaries. This will write train-test splits in the data folder.
2. Run ```src/get_predictive_latents.py -is_9b <is_9b> -layer <layer>``` to get predictive latents (```-is_9b 0 -layer 12``` for gemma-2b and ```-is_9b 1 -layer 20``` for gemma-9b)

## Steering
1. You will first need to generate brief hospital courses by completing only the first step (Process the MIMIC-IV Summaries) in the preprocessing pipeline [here](https://github.com/stefanhgm/patient_summaries_with_llms/tree/main/preprocess). The csv generated will be your input to the code below.
2. You will also need ``admissions.csv.gz`` from the [MIMIC-IV Physionet repo](https://physionet.org/content/mimiciv/3.1/hosp/).
3. Run ```src/steer_bhcs.py -bhcs_path <bhcs_generated_above> -admissions_path <mimic_iv_path> -race <race> -is_9b <is_9b> -factor <steering_factor>```

## Vignette race ablation
Run ```python ablate_race_vignette.py -is_9b <is_9b> -task <task> -temperature <temperature>```. ```task``` refers to the clinical condition (cocaine abuse, gestational hypertension, uterine fibroids).

## Clinical tasks

### Data preprocessing
#### Diagnosis Evidence
1. You will need the brief hospital courses file generated (see step 1 in Steering). Since we sampled BHCs and edited them to add race, we provide the ```note_ids``` as well as the text indices were race was inserted in the ```data``` folder. These will be used in the preprocessing code below.
2. Run ```python src/preprocess_bhcs.py -bhcs_path <bhcs_generated_above>```

 #### Q-Pain data preprocessing
 1. Save [Q-Pain](https://www.physionet.org/content/q-pain/1.0.0/) repository files in the ```data``` folder.
 2. Run ```python src/preprocess_qpain.py```

### Code
1. Run ```python src/ablate_race_bhc.py -is_9b <is_9b> -task <task> -multiple <multiple>```. ```multiple``` is set to 0 if you want to ablate only the middle layer, and 1 if you want to ablate 5 layers.
   



