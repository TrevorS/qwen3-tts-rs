.PHONY: lint fmt pre-commit pre-commit-install

lint: pre-commit

fmt:
	cargo fmt --all
	uvx ruff check --fix scripts/
	uvx ruff format scripts/

pre-commit-install:
	uvx pre-commit install

pre-commit:
	uvx pre-commit run --all-files
