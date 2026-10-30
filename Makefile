.PHONY: help install test data train eval clean

help:
	@echo "Available commands:"
	@echo "  make install  : Install dependencies and package"
	@echo "  make test     : Run the physics solver unit tests"
	@echo "  make data     : Generate the complex128 datasets"
	@echo "  make train    : Train the FNO model"
	@echo "  make eval     : Run evaluation (ID and OOD)"
	@echo "  make clean    : Remove generated data and results"

install:
	pip install torch numpy matplotlib pyyaml pytest tqdm
	pip install -e .

test:
	pytest tests/test_solver.py -v

data:
	python scripts/generate_data.py --config configs/data.yaml --out data/

train:
	python scripts/train.py --config configs/model_fno.yaml --data data/ --mode single --seed 42

eval:
	python scripts/evaluate.py --data data/ --ckpt results/fno_single_seed42.pt --split test_id
	python scripts/evaluate.py --data data/ --ckpt results/fno_single_seed42.pt --split test_ood

clean:
	rm -rf data/ results/ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +