NAME   = src
PYTHON = uv run python
FLAKE  = uv run flake8
MYPY   = uv run mypy
PDB    = uv run python -m pdb

all: run

run:
	@$(PYTHON) -m $(NAME)

cache:
	export UV_CACHE_DIR=/tmp/uv-cache
	export UV_PROJECT_ENVIRONMENT="/home/tsellak/goinfre/uv"

molicode:
	./moulinette/moulinette-ubuntu evaluate_student_search_results \
	data/output/search_results/UnansweredQuestions/dataset_code_public.json \
	data/datasets/AnsweredQuestions/dataset_code_public.json \
	--k 10

molidocs:
	./moulinette/moulinette-ubuntu evaluate_student_search_results \
	data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
	data/datasets/AnsweredQuestions/dataset_docs_public.json \
	--k 10

install:
	@pipx install uv
	@uv sync

debug:
	@$(PDB) -m $(NAME)

clean:
	@find . -type d -name "__pycache__" -print -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -print -exec rm -rf {} +

lint:
	@$(FLAKE) $(NAME)
	@$(MYPY) $(NAME) \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

a:
	@source .venv/bin/activate.fish

.PHONY: all run install lint lint-strict debug clean
