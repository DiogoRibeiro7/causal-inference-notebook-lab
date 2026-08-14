.PHONY: install install-docs test coverage lint format typecheck ci precommit \
	lab notebooks notebooks-all data docs docs-build clean

NOTEBOOKS_FAST = \
	notebooks/00_project_overview.ipynb \
	notebooks/01_confounding_propensity_scores.ipynb \
	notebooks/02_doubly_robust_aipw.ipynb

install:
	pip install -e ".[dev]"

install-docs:
	pip install -e ".[dev,docs]"

test:
	pytest

coverage:
	pytest --cov --cov-report=term-missing --cov-report=xml

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src

ci:
	make lint
	make typecheck
	make test

precommit:
	pre-commit run --all-files

lab:
	jupyter lab

data:
	python scripts/prepare_lalonde_job_training_dataset.py

notebooks:
	for nb in $(NOTEBOOKS_FAST); do \
		MPLBACKEND=Agg jupyter nbconvert --to notebook --execute $$nb --inplace; \
	done

# Executes every notebook in place, including the benchmark. Run `make data`
# first, or let notebook 10 download the dataset itself.
notebooks-all:
	for nb in notebooks/*.ipynb; do \
		MPLBACKEND=Agg jupyter nbconvert --to notebook --execute $$nb --inplace; \
	done

# Executes the notebooks into docs/notebooks (gitignored) and serves the site.
docs:
	python scripts/build_docs_notebooks.py
	mkdocs serve

docs-build:
	python scripts/build_docs_notebooks.py
	mkdocs build --strict

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .cache site docs/notebooks
	rm -rf build dist src/*.egg-info htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
