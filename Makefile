INPUT_FILE='churn/data/bank_data.csv'

clean:
	rm -Rf *.egg-info
	rm -Rf build
	rm -Rf dist
	rm -Rf .pytest_cache
	rm -f .coverage

build: clean
	python3 setup.py sdist

run: build
	python3 -m churn.main \
	--input_file $(INPUT_FILE)

tests:
	pytest -vv -s
	coverage run --source=churn -m pytest

linter:
	pylint churn --ignore-patterns=test --fail-under=7