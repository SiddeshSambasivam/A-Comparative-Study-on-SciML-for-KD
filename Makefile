.PHONY: clean env format run_dso_feynman

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = A-Comprehensive-Survey-on-SciML-for-KD
PYTHON_INTERPRETER = python

noise ?= 0.0
num_points ?= 0

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

feynman-aif:
	python -m src.benchmark -d data/AIFeynman/feynman_03.csv -n $(noise) -o logs/Feynman-03 -m aifeynman -p $(num_points)

feynman-gpl:
	python -m src.benchmark -d data/AIFeynman/feynman_03.csv -n $(noise) -o logs/Feynman-03 -m gplearn -p $(num_points)

feynman-dsr:
	python -m src.benchmark -d data/AIFeynman/feynman_03.csv -n $(noise) -o logs/Feynman-03 -m dsr -p $(num_points)

feynman-dsr-gp:
	python -m src.benchmark -d data/AIFeynman/feynman_03.csv -n $(noise) -o logs/Feynman-03 -m dsr-gp -p $(num_points)

feynman-nesymres:
	python -m src.benchmark -d data/AIFeynman/feynman_03.csv -n $(noise) -o logs/Feynman-03 -m nesymres -p $(num_points)

nguyen-aif:
	python -m src.benchmark -d data/Nguyen-12/nguyen-12.csv -n $(noise) -o logs/nguyen-12/ -m aifeynman -p $(num_points)

nguyen-gpl:
	python -m src.benchmark -d data/Nguyen-12/nguyen-12.csv -n $(noise) -o logs/nguyen-12/ -m gplearn -p $(num_points)

nguyen-dsr:
	python -m src.benchmark -d data/Nguyen-12/nguyen-12.csv -n $(noise) -o logs/nguyen-12/ -m dsr -p $(num_points)

nguyen-dsr-gp:
	python -m src.benchmark -d data/Nguyen-12/nguyen-12.csv -n $(noise) -o logs/nguyen-12/ -m dsr-gp -p $(num_points)

nguyen-nesymres:
	python -m src.benchmark -d data/Nguyen-12/nguyen-12.csv -n $(noise) -o logs/nguyen-12/ -m nesymres -p $(num_points)

sanity-check:
	python -m src.benchmark -d data/sanity_check/sanity_data.csv -n $(noise) -o logs/sanity_check/ -m $(model) -p $(num_points)

dash:
	streamlit run src/dashboard/main.py