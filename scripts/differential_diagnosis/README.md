# Lesion Differential Diagnosis based on Images and Expert's Features
The models developed in this project are trained based on lesion images and experts features.

## Efflorescence based differential diagnosis
### Data
Hand images of patients with eczema, lentigo, psoriasis and vitiligo.
Images without any efflorescences were labeled as healthy.
* Image directory: `/raid/dataset/diff_diags/hands_splitted_encrypted`
* Efflorescence labels: `/raid/dataset/diff_diags/hands_splitted_encrypted/labels_encrypted.p`

### Training
Training can be launched with the following command:
```
python /workspace/code/derm_pipeline/projects/differential_diagnosis/efflorescence_based.py \
        --encrypted \
        --data /workspace/data/diff_diags/hands_splitted_encrypted \
        --exp-name eff_based_dd --logdir /workspace/logs \
        --deterministic --full-precision
```
Example training logs: `/raid/logs/20220510_1741_resnet18_bs8_epo30_seed42_world1_normed_Adam_fp16_noCV_valid0.2_SL_deterministic__input512___img_classif_im-meta_effs_dd_eff_based_dd`

### Results
| Diagnosis | F1-score |
|-----------|----------|
| Eczema    | 0.65     |
| Healthy   | 1.00     |
| Lentigo   | 0.93     |
| Psoriasis | 0.68     |
| Vitiligo  | 0.87     |
| Average   | 0.83     |

## Location based differential diagnosis
### Data
Images of patients with Acne, Drug eruptions, Darier disease, Dyshidrotic eczema, Nummular dermatitis, Hand eczema, Impetigo, Melasma, Morphea, Onychomycosis, Palmoplantar keratoderma, Pityriasis rosea, Rosacea, Tinea pedis, Ulcer, Vitiligo.
* Image directory: `/raid/dataset/diff_diags/loc_with_diff_diags/localisation_diff_diags_auto_splitted_encrypted`
* Location labels: `/raid/dataset/diff_diags/loc_with_diff_diags/localisation_diff_diags_auto_splitted_encrypted/localisation_diff_diags_auto_splitted_encrypted_preds_df.p`

### Training
Training can be launched with the following command:
```
python /workspace/code/derm_pipeline/projects/differential_diagnosis/location_based.py \
        --encrypted \
        --data /workspace/data/diff_diags/loc_with_diff_diags/localisation_diff_diags_auto_splitted_encrypted \
        --exp-name loc_based_dd --logdir /workspace/logs --deterministic --full-precision --full-data
```
Example training logs: `/raid/logs/20220511_0850_resnet34_bs32_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input512___img_classif_im-meta_loc_dd_loc_based_dd`

### Results
| Diagnosis                | F1-score |
|--------------------------|----------|
| Acne                     | 0.82     |
| Drug eruptions           | 0.82     |
| Darier disease           | 0.43     |
| Dyshidrotic eczema       | 0.88     |
| Nummular dermatitis      | 0.89     |
| Hand eczema              | 0.79     |
| Impetigo                 | 0.78     |
| Melasma                  | 0.65     |
| Morphea                  | 0.88     |
| Onychomycosis            | 0.89     |
| Palmoplantar keratoderma | 0.84     |
| Pityriasis rosea         | 0.79     |
| Rosacea                  | 0.81     |
| Tinea pedis              | 0.77     |
| Ulcer                    | 0.92     |
| Vitiligo                 | 0.63     |
| Average                  | 0.79     |
