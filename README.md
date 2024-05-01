# Dermatology Deep Learning Pipeline
This pipeline is presented in the thesis "Deep learning in Clinical Dermatology" available on the [edoc server](https://edoc.unibas.ch/94022/) of the University of Basel.

The first goal of this pipeline is to simplify DL training procedures and enable dermatologists to ellaborate
their own models. 

The main data source for DL applications in dermatology being patient images, the field benefits from all
types of vision applications: classification, segmentation, object detection, etc.
Since they all train over images, there are many techniques, procedures and optimizations that resemble each others.
The second goal of this pipeline is to create an unified and extendable codebase for the different types of tasks.
To achieve this, we leverage the Fastai library which offers an uniform dataloading and training experience.

The pipeline is developped in OOP style in order to allow for code reuse and extensions for more specific tasks.
It also offers scripts useful for data preparation, model inference and results analysis.

## Requirements
The following libraries are needed alongside their dependencies:
* fastai
* tensorboard and tensorboardx
* opencv
* pillow 
* matplotlib
* numpy
* bidict
* efficientnet-pytorch

The following libraries are needed for object detection (still experimental):
* pycocotools
* icevision 5.X
* shapely

## Datasets format
The pipeline assumes the following data formats.
The examples show 3 datasets: weakly labeled data (wl_train), strongly labeled data (sl_train and sl_test).
There can be more than 3 datasets
### Classification
Every datasets should have all class directories. 
```
classification_data
+-- sl_test
│   +-- cls1
│   │   \-- img.jpg
│   +-- cls2
│   │   \-- img.jpg
│   \-- cls3
│       \-- img.jpg
+-- sl_train
│   +-- cls1
│   │   \-- img.jpg
│   +-- cls2
│   │   \-- img.jpg
│   \-- cls3
│       \-- img.jpg
\-- wl_train
    +-- cls1
    │   \-- img.jpg
    +-- cls2
    │   \-- img.jpg
    \-- cls3
        \-- img.jpg
```
### Segmentation
Images and masks should have the same name. Mask extension should be png.
```
segmentation_data
+-- sl_test
│   +-- images
│   │   \-- img.jpg
│   \-- masks
│       \-- img.png
+-- sl_train
│   +-- images
│   │   \-- img.jpg
│   \-- masks
│       \-- img.png
\-- wl_train
    +-- images
    │   \-- img.jpg
    \-- masks
        \-- img.png
```
### Object Detection
```
object_detection_data
+-- annotations
│   +-- sl_test.json
│   +-- sl_train.json
│   \-- wl_train.json
\-- images
    +-- sl_test
    │   \-- img.jpg
    +-- sl_train
    │   \-- img.jpg
    \-- wl_train
        \-- img.jpg
```
## Usage

### Classification
```
python src/training/distributed_launch.py src/classification/train_classification.py \
    --data classification_data --wl-train wl_train --sl-train sl_train --sl-tests sl_test \
    --input-size 32 --bs 64 --exp-name classif_demo --epochs 1 --fepochs 1 --lr .0002
```
### Segmentation
Images and masks should have the same name. Mask extension should be png.
```
python src/training/distributed_launch.py src/segmentation/train_segmentation.py \
    --data segmentation_data --wl-train wl_train --sl-train sl_train --sl-tests sl_test \
    --input-size 32 --bs 64 --exp-name segm_demo --epochs 1 --fepochs 1 --lr .0002
```
### Object Detection
```
python src/training/distributed_launch.py src/object_detection/train_object_detection.py \
    --data object_detection_data --wl-train wl_train --sl-train sl_train --sl-tests sl_test \
    --input-size 128 --bs 64 --exp-name od_demo --epochs 1 --fepochs 1 --lr .0002
```

## Dermatology Deep Learning Pipeline Applications
This directory contains the main applications of the dermatology pipeline.

### Reproducing experiments

#### Running environment
The docker image can be built using `projects/Dockerfile` and the following command:
```
docker build --pull --no-cache --tag fastai2 -f projects/Dockerfile .
```
All training commands should be run inside a docker container that can be created as follo:
```
docker run --gpus '"device=4"' -it --rm -v /raid/dataset:/workspace/data -v /raid/code:/workspace/code -v /raid/logs:/workspace/logs --ipc=host --name test_pipeline fastai2:latest
```

#### Experiment logging and results
All training commands will create a log directory (usually in /raid/logs) that contains at least three subdirectories:
* Model weights
* Tensorboard logs recording losses and validation metrics
* Test performance plots. One directory will be created per test set.

The test performance are also printed on stdout at the end of the training.
