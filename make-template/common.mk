# Copied from https://github.com/adepretis/docker-make-stub/blob/master/common.mk
RED       := $(shell tput -Txterm setaf 1)
GREEN     := $(shell tput -Txterm setaf 2)
YELLOW    := $(shell tput -Txterm setaf 3)
BLUE      := $(shell tput -Txterm setaf 4)
MAGENTA   := $(shell tput -Txterm setaf 5)
CYAN      := $(shell tput -Txterm setaf 6)
WHITE     := $(shell tput -Txterm setaf 7)
RESET     := $(shell tput -Txterm sgr0)

SMUL      := $(shell tput smul)
RMUL      := $(shell tput rmul)

# Add the following 'help' target to your Makefile
# And add help text after each target name starting with '\#\#'
# A category can be added with @category
HELP_FUN = \
	%help; \
	use Data::Dumper; \
	while(<>) { \
		if (/^([_a-zA-Z0-9\-%]+)\s*:.*\#\#(?:@([a-zA-Z0-9\-_\s]+))?\t(.*)$$/ \
			|| /^([_a-zA-Z0-9\-%]+)\s*:.*\#\#(?:@([a-zA-Z0-9\-]+))?\s(.*)$$/) { \
			$$c = $$2; $$t = $$1; $$d = $$3; \
			push @{$$help{$$c}}, [$$t, $$d, $$ARGV] unless grep { grep { grep /^$$t$$/, $$_->[0] } @{$$help{$$_}} } keys %help; \
		} \
	}; \
	for (sort keys %help) { \
		printf("${WHITE}%24s:${RESET}\n\n", $$_); \
		for (@{$$help{$$_}}) { \
			printf("%s%25s${RESET}%s   %s${RESET}\n", \
				( $$_->[2] eq "Makefile" || $$_->[0] eq "help" ? "${YELLOW}" : "${WHITE}"), \
				$$_->[0], \
				( $$_->[2] eq "Makefile" || $$_->[0] eq "help" ? "${GREEN}" : "${WHITE}"), \
				$$_->[1] \
			); \
		} \
		print "\n"; \
	}

# make
.DEFAULT_GOAL := help

# Variable wrapper
define defw
	custom_vars += $(1)
	$(1) ?= $(2)
	export $(1)
	shell_env += $(1)="$$($(1))"
endef

# Variable wrapper for hidden variables
define defw_h
	$(1) := $(2)
	shell_env += $(1)="$$($(1))"
endef

# Colorized output for control functions (info, warning, error)
define verbose_info
	@echo ${CYAN}
	@echo ================================================================================
	@echo ${1}
	@echo ================================================================================
	@echo
	@tput -Txterm sgr0 # ${RESET} won't work here for some reason
endef

define verbose_warning
	@echo ${YELLOW}
	@echo ================================================================================
	@echo ${1}
	@echo ================================================================================
	@echo
	@tput -Txterm sgr0 # ${RESET} won't work here for some reason
endef

define verbose_error
	@echo ${RED}
	@echo ================================================================================
	@echo ${1}
	@echo ================================================================================
	@echo
	@tput -Txterm sgr0 # ${RESET} won't work here for some reason
endef

define HELPTEXT
	@echo ""
	@printf "%30s   %-100s\n" "${WHITE}Example usage" "${MAGENTA}make test GPU='false'"
	@echo ""
	@printf "%30s\n\n" "${CYAN}VARIABLES"
	@printf "%30s   %-100s\n" "${WHITE}CONTAINER_NAME" "set the name of the docker container, default is 'default'"
	@printf "%30s   %-100s\n" "${WHITE}DOCKER_SRC_DIR" "source directory within the docker container, default is '/app/'"
	@printf "%30s   %-100s\n" "${WHITE}LOCAL_DATA_DIR" "set the LOCAL_DATA_DIR variable to the path of the dataset, default is the path on the GPU '/data/'"
	@printf "%30s   %-100s\n" "${WHITE}GPU" "set the GPU variable to run the targets with ('true') or without ('false') GPU support, default is 'true'"
	@printf "%30s   %-100s\n" "${WHITE}GPU_ID" "set the GPU id you want to run on (comma separated list of indexes), empty and GPU is 'true', all GPUs will be used"
	@printf "%30s   %-100s\n" "${WHITE}PORT" "set the external port redirected, default to 8888 if available otherwise either 9999 or 10000"
	@echo "${RESET}"
	@echo ""
	@printf "%30s" "${CYAN}TARGETS"
	@echo "${RESET}"
	@echo ""
	@perl -e '$(HELP_FUN)' $(MAKEFILE_LIST)
endef

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


.PHONY: help
help: ##@Usage show this help
	$(HELPTEXT)
