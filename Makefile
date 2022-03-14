SHELL=/bin/bash

PACAKGE = black mypy isort flake8 python-box bbox-utility pytorch-lightning timm==0.5.4 torchmetrics

setup: ## setup package on kaggle docker image
	python --version \
	&& python -m pip install --upgrade pip \
	&& python -m pip install poetry \
	&& poetry install \
	&& poe force-cuda11

set_tpu:
	${POETRY} \
	&& poetry config virtualenvs.create false --local \
	&& poetry install \
	&& poetry run python3 -m pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

pip_export:
	pip3 freeze > requirements.txt

develop_by_requirements:
	for package in $(cat requirements.txt); do poetry add "${package}"; done

download_datasets: ## download datasets (you need kaggle-api)
	python ./src/tasks/fetch_datasets.py --compe_name happy-whale-and-dolphin

update_datasets:
	zip -r output/sub.zip output/sub
	kaggle datasets version -p ./output/sub -m "Updated data"

pull_kaggle_image:
	docker pull gcr.io/kaggle-gpu-images/python

build_dev_image:
	docker build -f Dockerfile -t local-kaggle-python .
