.PHONY: lint fmt pre-commit pre-commit-install profile-chrome profile-flamegraph profile-nsys audit-gpu-syncs test-kernel count-kernels

MODEL_DIR ?= test_data

lint: pre-commit

fmt:
	cargo fmt --all
	uvx ruff check --fix scripts/
	uvx ruff format scripts/

pre-commit-install:
	uvx pre-commit install

pre-commit:
	uvx pre-commit run --all-files

# ── Profiling ────────────────────────────────────────────────────────────

profile-chrome:
	cargo run --profile=profiling --features=profiling,cuda,cli --bin e2e_bench -- \
		--model-dir $(MODEL_DIR) --iterations 1
	@echo "Trace written to trace.json — open in chrome://tracing or https://ui.perfetto.dev"

profile-flamegraph:
	cargo flamegraph --profile=profiling --features=cuda,cli --bin e2e_bench -- \
		--model-dir $(MODEL_DIR) --iterations 1

profile-nsys:
	nsys profile --trace=cuda,nvtx --output=nsys_report \
		cargo run --profile=profiling --features=cuda,cli --bin e2e_bench -- \
			--model-dir $(MODEL_DIR) --iterations 1

audit-gpu-syncs:
	@bash scripts/audit-gpu-syncs.sh

# ── Kernel Development ──────────────────────────────────────────────────

test-kernel:
	@bash scripts/test-kernel.sh $(NAME)

count-kernels:
	@bash scripts/count-kernels.sh $(MODEL_DIR)
