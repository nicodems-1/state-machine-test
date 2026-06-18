MAIN = src/main.py
OUTPUT = data/output
PYTHON_TEMP = src/__pycache__
MYPY_TEMP = .mypy_cache

init-data:
	mkdir -p data/output
	touch data/output/output.json

install:
	uv sync 

run: install init-data
	uv run python $(MAIN)	

debug: install
	uv pdbg python $(MAIN)

clean:
	rm -rf $(OUTPUT) $(PYTHON_TEMP) $(MYPY_TEMP)

lint: install
	uv run flake8 src ;\
	uv run mypy src \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

.PHONY: install run debug clean lint init-data