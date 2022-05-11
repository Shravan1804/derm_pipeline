# Anatomy Mapping of Patients' Photographs
The models aim to map anatomical regions in patients' photographs.

## Macro-anatomy mapping
### Data
Dataset composed of strongly labeled images and unlabeled images of the main body regions: arms, legs, feet, hands, head, other and trunk.
* Image root directory: `/raid/dataset/anatomy_project/body_loc/USZ_pipeline_cropped_images_patched_512_encrypted`
* Unlabeled images: `weak_labels`
* Strongly labeled images: `strong_labels_train`, `strong_labels_test`, `strong_labels_test_balanced510`

### Training
Creation of the docker container:
```
docker run --gpus '"device=4"' -it --rm -v /raid/dataset:/workspace/data -v /raid/code:/workspace/code -v /raid/logs:/workspace/logs --ipc=host --name test_pipeline fastai2:latest
```
Training can be launched with the following command:
```
python /workspace/code/derm_pipeline/projects/anatomy/body_loc.py \
        --encrypted  \
        --data /workspace/data/anatomy_project/body_loc/USZ_pipeline_cropped_images_patched_512_encrypted \
        --sl-train strong_labels_train --sl-tests strong_labels_test_balanced510 strong_labels_test \
        --progr-size --exp-name body_loc --logdir /workspace/logs --deterministic
```
Example training logs: `/raid/logs/20220511_1047_efficientnet-b2_bs32_epo30_seed42_world1_normed_Adam_fp16_noCV_valid0.2_SL_deterministic__input260_progr-size0.5_0.75_1___img_classif_body_loc`

### Results
| Diagnosis | F1-score |
|-----------|----------|
| Arms      | 0.00     |
| Legs      | 0.00     |
| Feet      | 0.00     |
| Hands     | 0.00     |
| Head      | 0.00     |
| Other     | 0.00     |
| Trunk     | 0.00     |
| Average   | 0.00     |

## Micro-anatomy mapping
### Data
Images of specific body regions with semantic labels of the different anatomical subregions.
* Image root directory: `/raid/dataset/anatomy_project`
* Ear images: `ear_anatomy/ear_segm_splitted_encrypted`
* Eye images: `eye_anatomy/eye_segm_splitted_encrypted`
* Hand images: `hands_anatomy/hands_segm_splitted_encrypted`
* Mouth images: `mouth_anatomy/before_correction/mouth_segm_splitted_encrypted` (note: student is expected to label additional pictures and correct labels)
* Nail images: `nail_anatomy/nail_segm_splitted_encrypted`

### Training
Creation of the docker container:
```
docker run --gpus '"device=4"' -it --rm -v /raid/dataset:/workspace/data -v /raid/code:/workspace/code -v /raid/logs:/workspace/logs --ipc=host --name test_pipeline fastai2:latest
```
Training can be launched with the following command (replacing the --region and -exp-name parameter for the different regions, namely ear, eye, hand, mouth, nail or all):
```
python /workspace/code/derm_pipeline/projects/anatomy/segm_anato.py \
        --encrypted --data /workspace/data/anatomy_project \
        --logdir /workspace/logs --deterministic \
        --full-data --exp-name hand --region hand
```
Example training logs:
* Ear images: `/raid/logs/20220511_1158_resnet50_bs4_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input380__img_segm_ear-anato_ear`
* Eye images: `/raid/logs/20220511_1159_resnet50_bs4_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input380__img_segm_eye-anato_eye`
* Hand images: `/raid/logs/20220511_1157_resnet50_bs4_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input380__img_segm_hand-anato_hand`
* Mouth images: `/raid/logs/20220511_1200_resnet50_bs4_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input380__img_segm_mouth-anato_mouth`
* Nail images: `/raid/logs/20220511_1200_resnet50_bs4_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input380__img_segm_nail-anato_nail`
* All images: `/raid/logs/20220511_1311_resnet50_bs4_epo30_seed42_world1_normed_Adam_fp16_fullData_SL_deterministic__input380__img_segm_all-anato_all`

### Results
In the case of the eye region, labels include the abnormal semantic class that should be ignored in performance (class used only for few images).

| Diagnosis        | F1-score |
|------------------|----------|
| Average ear      | 0.80     |
| Average eye      | 0.83     |
| Average hand     | 0.83     |
| Average mouth    | 0.66     |
| Average nail     | 0.65     |
| Average all      | -        |
