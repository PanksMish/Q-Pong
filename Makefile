.PHONY: install test smoke data train experiments tables figures all

install:
	pip install -e ".[all]"

test:
	pytest -q

smoke:
	python scripts/smoke_test.py ieee30

data:
	python scripts/prepare_data.py --rebuild

train:
	python scripts/train_pinn.py --variant qpong data_only pinn_opf --n-jobs $${N_JOBS:-8}

experiments:
	bash scripts/run_all.sh

tables:
	python scripts/make_tables.py

figures:
	python scripts/make_figures.py
