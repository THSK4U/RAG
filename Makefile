NAME   = src
PYTHON = uv run python
FLAKE  = uv run flake8
MYPY   = uv run mypy
PDB    = uv run python -m pdb

all: run

run:
	@$(PYTHON) -m $(NAME)
	
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


.PHONY: all run install lint lint-strict debug clean
