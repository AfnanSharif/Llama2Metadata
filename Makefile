install:
	./scripts/setup.sh

test:
	python3 -m unittest discover -s tests

lint:
	./scripts/lint.sh

clean:
	rm -rf __pycache__ .pytest_cache
