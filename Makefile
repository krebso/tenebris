black:
	black .

flake8:
	flake8 .

isort:
	isort --settings-file pyproject.toml .

mypy:
	mypy --config-file ./pyproject.toml --install-types --non-interactive --namespace-packages --explicit-package-bases tenebris/

pycln:
	pycln --config pyproject.toml .

install_precommit:
	pre-commit install && pre-commit autoupdate

uninstall_precommit:
	pre-commit uninstall

check:
	$(MAKE) pycln black isort flake8 mypy
