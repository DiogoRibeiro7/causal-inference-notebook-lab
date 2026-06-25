.PHONY: install test lint typecheck ci lab notebooks clean

NOTEBOOKS_FAST = \
	notebooks/00_project_overview.ipynb \
	notebooks/01_confounding_propensity_scores.ipynb \
	notebooks/02_doubly_robust_aipw.ipynb

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

ci:
	make lint
	make typecheck
	make test

lab:
	jupyter lab

notebooks:
	for nb in $(NOTEBOOKS_FAST); do \
		MPLBACKEND=Agg jupyter nbconvert --to notebook --execute $$nb --inplace; \
	done

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
