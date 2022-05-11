# Dermatology Deep Learning Pipeline Applications
This directory contains the main applications of the dermatology pipeline.

## Reproducing experiments
### Data
The provided datasets paths are for the DGX machine. 

### Running environment
The docker image can be built using `projects/Dockerfile` and the following command:
```
docker build --pull --no-cache --tag fastai2 -f projects/Dockerfile .
```
All training commands should be run inside a docker container that can be created as follo:
```
docker run --gpus '"device=4"' -it --rm -v /raid/dataset:/workspace/data -v /raid/code:/workspace/code -v /raid/logs:/workspace/logs --ipc=host --name test_pipeline fastai2:latest
```

### Experiment logging and results
All training commands will create a log directory (usually in /raid/logs) that contains at least three subdirectories:
* Model weights
* Tensorboard logs recording losses and validation metrics
* Test performance plots. One directory will be created per test set.

The test performance are also printed on stdout at the end of the training.