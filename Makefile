.PHONY: install install-cuda install-cuda-full install-dev install-all download-checkpoints lint format typecheck test test-cov train evaluate sweep hpo ablation benchmark quick-pilot analyze figures reproduce clean

# === Environment ===
install:
	uv sync

install-cuda:
	uv sync --extra cuda

# Full CUDA install including mamba-ssm/causal-conv1d.
# Requires system CUDA toolkit matching the PyTorch CUDA version.
# Falls back to PyTorch-native ops if prebuilt wheels are unavailable.
install-cuda-full:
	CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=TRUE \
		uv sync --extra cuda-mamba

install-dev:
	uv sync --extra dev

install-all:
	CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=TRUE \
		uv sync --extra cuda-mamba --extra dev

# === Pretrained checkpoints ===
# SSAMBA-tiny AudioSet checkpoint (Shams et al., https://github.com/SiavashShams/ssamba),
# mirrored on Hugging Face. Override SSAMBA_CKPT_URL to use a different source.
SSAMBA_CKPT_URL ?= https://huggingface.co/attentionisallyouneed369/ssamba/resolve/main/ssamba_tiny_400.pth

download-checkpoints:
	mkdir -p checkpoints
	curl -fL --retry 3 --retry-all-errors "$(SSAMBA_CKPT_URL)" -o checkpoints/ssamba_tiny_400.pth

# === Code Quality ===
lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

# === Testing ===
test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=bioaed --cov-report=html

# === Training & Evaluation ===
train:
	uv run python -m bioaed.train

evaluate:
	uv run python -m bioaed.evaluate

sweep:
	uv run python -m bioaed.sweep

hpo:
	uv run python -m bioaed.sweep \
		model=inceptiontime,ast,audio_mamba \
		dataset=aswine,anuraset \
		n_trials=30 --multirun

# === Ablation & Analysis ===
quick-pilot:
	bash scripts/quick_experiment.sh

benchmark:
	bash scripts/run_benchmark.sh

ablation:
	uv run python -m bioaed.ablation --multirun \
		model=inceptiontime,ast,audio_mamba \
		dataset=aswine,anuraset \
		seed=0,1,2,3,4

analyze:
	uv run python -m bioaed.evaluation.report_generator

figures:
	uv run python -m bioaed.evaluation.report_generator

# Full end-to-end reproduction of the paper (GPU-intensive; see the script header).
reproduce:
	bash scripts/reproduce_paper.sh

# === Cleanup ===
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
	rm -rf outputs/ multirun/
