black:
	black .

flake8:
	flake8 .

isort:
	isort --settings-file pyproject.toml .

mypy:
	mypy --config-file pyproject.toml --install-types --non-interactive --namespace-packages --explicit-package-bases .

pycln:
	pycln --config pyproject.toml .

test:
	pytest --config pyproject.toml .

check:
	$(MAKE) pycln black isort flake8 mypy
