NAME   = src
PYTHON = uv run python
FLAKE  = uv run flake8
MYPY   = uv run mypy
PDB    = uv run python -m pdb

all: run

run:
	@$(PYTHON) -m $(NAME)

cache:
	@echo "Linking to goinfre..."
	@mkdir -p /home/tsellak/goinfre/uv /home/tsellak/goinfre/huggingface /home/tsellak/goinfre/rag_venv
	@rm -rf ~/.cache/uv ~/.cache/huggingface .venv
	@ln -sfn /home/tsellak/goinfre/uv ~/.cache/uv
	@ln -sfn /home/tsellak/goinfre/huggingface ~/.cache/huggingface
	@ln -sfn /home/tsellak/goinfre/rag_venv .venv
	@uv sync
	@echo "Cache and venv successfully linked to goinfre!"

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
