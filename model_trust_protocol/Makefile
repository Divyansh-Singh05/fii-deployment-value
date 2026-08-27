.PHONY: help install lint typecheck test channels verify table1 all clean

PY ?= uv run

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## create the environment and install the package
	uv venv --python 3.13
	uv pip install -e ".[dev]"

lint:  ## style and static checks
	$(PY) ruff check src tests
	$(PY) ruff format --check src tests

typecheck:  ## strict type checking
	$(PY) mypy src

test:  ## unit tests
	$(PY) pytest

applications:  ## list registered applications and where each dies
	$(PY) convgap applications

verify:  ## trace every number to its source artifact
	$(PY) convgap verify

table1:  ## build the conversion exhibit
	$(PY) convgap table1

access:  ## build the IEEE Access paper and compile PDF
	python3 scripts/build_access_tex.py

all: lint typecheck test applications  ## everything that does not need the source trees

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs/exhibits/*.csv
