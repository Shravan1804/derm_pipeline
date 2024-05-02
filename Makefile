###########################
# VARIABLES
###########################
PROJECT_NAME := derm_pipeline
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)

include make-template/*.mk

WANDB_CACHE_DIRS = -v $$PWD/wandb_config_dir:/.config/ -v $$PWD/wandb_netrc_dir:/.netrc/ -v $$PWD/wandb_cache_dir:/.cache
DOCKER_ARGS := -v $(HOME):/workspace/code -v $(LOCAL_DATA_DIR):/data/ -v $$PWD/numba_cache_dir:/numba_cache_dir/ -v $$PWD/mpl_cache_dir:/mpl_cache_dir/ $(WANDB_CACHE_DIRS)  -p $(PORT):8888 --rm --shm-size=4G
DOCKER_CMD := docker run $(DOCKER_ARGS) --env-file=.env $(GPU_ARGS) $(DOCKER_CONTAINER_NAME) -it $(PROJECT_NAME):$(GIT_BRANCH)

###########################
# PROJECT UTILS
###########################
.PHONY: clean
clean:  ##@Utils clean the project
	@black .
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f .coverage
	@rm -f core

.PHONY: create_cache_dirs
create_cache_dirs: ##@Create a cache directories, librosa, matplotlib, wandb config
	mkdir -p ./numba_cache_dir; mkdir -p ./mpl_cache_dir; mkdir -p ./wandb_config_dir; mkdir -p ./wandb_netrc_dir; mkdir -p ./wandb_cache_dir; mkdir -p ./wandb

.PHONY: test
test: _build  ##@Utils run all tests in the project (unit and integration tests)
	$(DOCKER_CMD) /bin/bash -c "python3 -m coverage run -m pytest tests --durations=0 --junitxml=report.xml; coverage report -i --include=src/*"

.PHONY: unittest
unittest: _build  ##@Utils run all unit tests in the project
	$(DOCKER_CMD) /bin/bash -c "python3 -m coverage run -m pytest tests --junitxml=report.xml --ignore=tests/integration_tests; coverage report -i --include=src/*"

.PHONY: install
install:  ##@Utils install the dependencies for the project
	python3 -m pip install -r requirements.txt

.PHONY: install_dev
install_dev:  ##@Utils install the development dependencies for the project
	python3 -m pip install -r requirements_dev.txt

.PHONY: install_ci
install_ci:  ##@Utils install the CI dependencies for the project
	python3 -m pip install -r requirements_ci.txt

.PHONY: generate_requirements
generate_requirements: ##@Utils to generate requirements.txt from the src/ and test/ directories
	pipreqs --savepath requirements_pipreqs.txt --debug src; pipreqs --savepath requirements_pipreqs_test.txt --debug test;\
	  sort -u requirements_pipreqs.txt requirements_pipreqs_test.txt > requirements_proposal.txt;\
	  rm requirements_pipreqs.txt requirements_pipreqs_test.txt;

.PHONY: debug_port
debug_port:
	echo "$(PORT_USED)"
	echo "$(NEXT_TCP_PORT)"
	echo "$(PORT)"

###########################
# UTILS
###########################
.PHONY: _check_arguments
_check_arguments:
	@if [ -z $${ARGS+x} ]; then\
		echo "WARNING: Pass arguments with";\
		echo ">	make ... ARGS='-p DOWNLOAD_PATH'";\
		exit 1;\
	fi

###########################
# DOCKER
###########################
_build: create_cache_dirs
	@echo "Building image $(PROJECT_NAME):$(GIT_BRANCH)"
	@docker build -f Dockerfile -t $(PROJECT_NAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker run an interactive bash inside the docker image (default: GPU=true)
	@echo "Running bash with GPU being $(GPU) and GPU_ID $(GPU_ID)"
	$(DOCKER_CMD) /bin/bash; \

start_jupyter: _build  ##@Docker start a jupyter notebook inside the docker image (default: GPU=true)
	@echo "Starting jupyter notebook $(shell [ '$(GPU)' = 'true' ] && echo 'with GPUs $(GPU_ID)')"; \
	$(DOCKER_CMD) /bin/bash -c "jupyter notebook --allow-root --ip 0.0.0.0 --port $(PORT)"; \

###########################
# SCRIPTS
###########################
