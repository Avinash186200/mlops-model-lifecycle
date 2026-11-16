.PHONY: install test all pipeline serve docker

install:
	pip install -e ".[dev,serve]"

test:
	pytest -q

# Rebuilds every number in results/ and the docs (about 30 s on a laptop).
all:
	python -m mlcycle.cli run-all

# The small CI pipeline: tiny data -> train -> gate -> register.
pipeline:
	python -m mlcycle.cli pipeline --artifacts artifacts/ci

serve:
	python -m mlcycle.cli serve --registry artifacts/registry

docker:
	docker build -t mlcycle-api .
