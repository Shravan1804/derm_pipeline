###########################
# DOCKER ARGUMENTS
###########################
ifeq ($(origin CONTAINER_NAME), undefined)
  CONTAINER_NAME := default
  $(warning ${YELLOW}WARNING: CONTAINER_NAME is undefined. Set to '${CONTAINER_NAME}'.${RESET})
endif

ifeq ($(origin DOCKER_SRC_DIR), undefined)
  DOCKER_SRC_DIR := "/app/"
endif

ifeq ($(shell test -e .env && echo -n yes), yes)
  ENV_FILE := --env-file=.env
else
  ENV_FILE :=
endif

ifeq ($(origin LOCAL_DATA_DIR), undefined)
  LOCAL_DATA_DIR := /data/
  $(warning ${YELLOW}WARNING: LOCAL_DATA_DIR is undefined. Set to '${LOCAL_DATA_DIR}'.${RESET})
endif

ifeq ($(origin PROJECT_NAME), undefined)
  PROJECT_NAME := unknown
endif

ifeq ("$(GPU)", "false")
  GPU_ARGS :=
  DOCKER_CONTAINER_NAME := --name $(PROJECT_NAME)_$(CONTAINER_NAME)
  ifneq ($(origin GPU_ID), undefined)
    $(warning ${YELLOW}WARNING: GPU is set to false but GPU_ID is set to '${GPU_ID}'. Ignoring GPU_ID.${RESET})
  endif
else
  ifeq ($(origin GPU_ID), undefined)
    GPU_ID := all
    GPU_NAME := ${GPU_ID}
    $(warning ${YELLOW}WARNING: GPU_ID is undefined. Set to GPU_ID=${GPU_ID} and GPU_NAME=${GPU_NAME}.${RESET})
    NUM_GPUS := ${shell nvidia-smi -L | wc -l}
  else
    COMMA := ,
    DASH := -
    GPU_NAME = $(subst ${COMMA},${DASH},${GPU_ID})
    $(warning ${YELLOW}WARNING: Set GPU_NAME=${GPU_NAME}.${RESET})
    EMPTY :=
    SPACE := ${EMPTY} ${EMPTY}
    count = $(words $1)$(if $2,$(call count,$(wordlist 2,$(words $1),$1),$2))
    GPU_LIST := $(subst $(COMMA),$(SPACE),$(GPU_ID))
    NUM_GPUS := $(call count,$(GPU_LIST))
  endif

  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Linux)
    NUM_CORES := $(shell nproc)
  else ifeq ($(UNAME_S),Darwin)
    NUM_CORES := $(shell sysctl -n hw.ncpu)
  else
    NUM_CORES := $(NUM_GPUS)
  endif
  # optimal number of threads is #cores/#gpus
  NUM_THREADS := $(shell expr $(NUM_CORES) / $(NUM_GPUS))
  # NOTE:
  # For now we don't use the max number of threads, as we experienced some unexpected behaviour.
  # Set `OMP_NUM_THREADS=$(NUM_THREADS)` before starting the script in order to activate again.

  GPU_ARGS := --gpus '"device=$(GPU_ID)"' --shm-size 8G
  DOCKER_CONTAINER_NAME := --name $(PROJECT_NAME)_gpu_$(GPU_NAME)_$(CONTAINER_NAME)
endif

INTERACTIVE:=$(shell [ -t 0 ] && echo 1)
ifdef INTERACTIVE
  TTY =-it
else
  TTY =
endif

# check if `netstat` is installed
ifeq (, $(shell which netstat))
  $(error "Netstat executable not found, install it with `apt-get install net-tools`")
endif
# Check if Jupyter Port is already use and define an alternative
ifeq ($(origin PORT), undefined)
  PORT_USED = $(shell netstat -tln | grep -E '(tcp|tcp6)' | grep -Eo '8888' | tail -n 1)
  # Will fail if both ports 9999 and 10000 are used, I am sorry for that
  NEXT_TCP_PORT = $(shell netstat -tln | grep -E '(tcp|tcp6)' | grep -Eo '[0-9]{4}' | sort | tail -n 1 | xargs -I '{}' expr {} + 1)
  ifeq (${PORT_USED}, 8888)
    PORT = ${NEXT_TCP_PORT}
  else
    PORT = 8888
  endif
endif

CACHE_DIRS := -v $$PWD/numba_cache_dir:/numba_cache_dir/ -v $$PWD/mpl_cache_dir:/mpl_cache_dir/
DOCKER_ARGS := -v $$PWD:${DOCKER_SRC_DIR} -v ${LOCAL_DATA_DIR}:/data/ ${CACHE_DIRS} -p ${PORT}:${PORT} --rm
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD | tr / _)
DOCKER_CMD := docker run ${DOCKER_ARGS} ${ENV_FILE} ${GPU_ARGS} ${DOCKER_CONTAINER_NAME} ${TTY} ${PROJECT_NAME}:${GIT_BRANCH}
