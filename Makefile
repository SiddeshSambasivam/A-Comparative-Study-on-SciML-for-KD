.PHONY: clean env format run_dso_feynman

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = A-Comprehensive-Survey-on-SciML-for-KD
PYTHON_INTERPRETER = python

noise ?= 0.0

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

init:
## check if conda is install else ask user to install conda
ifeq ($(HAS_CONDA),False)
	echo "Conda is not installed. Please install conda."
	exit 1
endif

## init conda to whatever shell is being used
ifeq ($(HAS_CONDA),True)
	conda init $(shell which $(shell sh -c 'echo $SHELL'))
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

feynman:
	python -m src.benchmark -d data/AIFeynman/feynman_03.csv -n $(noise)

sanity_check:
	python -m src.benchmark -d data/sanity_check/sanity_data.csv -n $(noise)