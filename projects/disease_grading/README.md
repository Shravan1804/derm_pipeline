# Automation of Disease Severity Grading
The models in this project aim to segment lesions in patients' photographs.

## Hand Eczema
### Data
Images semantic labeled for classes other, skin, eczema.
* Image directory: `/raid/dataset/disease_grading/eczema/eczema_splitted_patched_512_encrypted`

### Training
Training can be launched with the following command:
```
python /workspace/code/derm_pipeline/projects/disease_grading/he.py \
        --encrypted \
        --data /workspace/data/disease_grading/eczema/eczema_splitted_patched_512_encrypted \
        --exp-name he --logdir /workspace/logs --deterministic --full-data
```
Example training logs: `/raid/logs/20220511_1329_resnet18_bs16_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input256__img_segm_he`

### Results
| Class   | F1-score |
|---------|----------|
| Other   | 1.00     |
| Skin    | 0.95     |
| Eczema  | 0.70     |
| Average | 0.88     |

## Ichthyosis with confetti
### Data
Images semantic labeled for classes other, white_skin, non_white_skin.
* Image directory: `/raid/dataset/disease_grading/iwc/segm_iwc_with_objs_splitted_patched_512_encrypted`

### Training
Training can be launched with the following command:
```
python /workspace/code/derm_pipeline/projects/disease_grading/iwc.py \
        --encrypted \
        --data /workspace/data/disease_grading/iwc/segm_iwc_with_objs_splitted_patched_512_encrypted \
        --exp-name iwc --logdir /workspace/logs --deterministic --full-data
```
Example training logs: `/raid/logs/20220510_1657_resnet18_bs16_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input256__img_segm_iwc`

### Results
| Class          | F1-score |
|----------------|----------|
| Other          | 0.96     |
| White skin     | 0.80     |
| Non white skin | 0.94     |
| Average        | 0.90     |

## Palmoplantar pustular psoriasis
### Data
Images semantic labeled for classes other, skin, pustules, spots. The skin label is discarded for lesion segmentation training.
* Image directory: `/raid/dataset/disease_grading/ppp_grading/PPP_whs_corr_wskin_splitted_patched_512_encrypted`
* Cropped train images: Additional train directory generated with `segmentation/crop_to_thresh.py`, which crops images around lesions

### Training
Training can be launched with the following command:
```
python /workspace/code/derm_pipeline/projects/disease_grading/ppp.py \
        --encrypted \
        --data /workspace/data/disease_grading/ppp_grading/PPP_whs_corr_splitted_no_patient_leak_patched_512_encrypted \
        --sl-train train train_cropped --exp-name ppp_with_crops
        --logdir /workspace/logs --deterministic --focal-loss-plus-dice-focal-loss --full-data
```
Example training logs: `/raid/logs/20220428_1845_resnet50_bs8_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input380_focal-plus-dice_focal-loss__img_segm_ppp_ppp_with_crops`

### Results
| Class       | F1-score |
|-------------|----------|
| Other       | 1.00     |
| Pustules    | 0.66     |
| Brown spots | 0.65     |
| Average     | 0.77     |

