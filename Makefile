.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := derm_pipeline
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
# docker
DOCKER_CMD := docker run -v $$PWD:/workspace/ --shm-size 8G -it $(PROJECTNAME):$(GIT_BRANCH)
DOCKER_GPU_CMD := docker run -v $$PWD:/workspace/ --gpus='"device=0"' --shm-size 8G -it $(PROJECTNAME):$(GIT_BRANCH)

###########################
# COMMANDS
###########################
# Thanks to: https://stackoverflow.com/a/10858332
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

###########################
# PROJECT UTILS
###########################
.PHONY: clean
clean:  ##@Utils cleanes the project
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f .coverage
	@rm -f core

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker runs an interacitve bash inside the docker image
	@echo "Run inside docker image"
	$(DOCKER_CMD) /bin/bash

run_gpu_bash: _build  ##@Docker runs an interacitve bash inside the docker image with a GPU
	@echo "Run inside docker image"
	$(DOCKER_GPU_CMD) /bin/bash

###########################
# TRAINING
###########################
train_bodyloc: ##@Training trains body localization
	docker run --gpus '"device=6,7"' -it --rm -v /raid/dataset:/workspace/data -v /raid/code:/workspace/code -v /raid/logs:/workspace/logs --ipc=host --name body_loc fastai2:latest
	python /workspace/code/derm_pipeline/training/distributed_launch.py --encrypted /workspace/code/derm_pipeline/projects/anatomy/body_loc.py --data /workspace/data/anatomy_project/body_loc/USZ_pipeline_cropped_images_patched_512_encrypted --sl-train strong_labels_train --sl-tests strong_labels_test_balanced510 --progr-size --exp-name body_loc_lr.002 --logdir /workspace/logs --reduce-lr-on-plateau --deterministic --lr .002
