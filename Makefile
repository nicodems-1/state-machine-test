MAIN = src/main.py
OUTPUT = data/output.json

init-data:
	mkdir -p data/output
	touch data/output.json

install:
	uv sync 

run: install init-data
	uv run python $(MAIN)	

debug: install
	uv pdbg python $(MAIN)

clean:
	rm -rf $(OUTPUT) .venv

lint: install
	uv run flake8 src
	uv run mypy . \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

.PHONY: install run debug clean lint init-data