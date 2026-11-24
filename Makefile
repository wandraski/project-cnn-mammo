.PHONY: test smoke lint scan clean

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

compile:
	python -m compileall -q src tests

smoke:
	PYTHONPATH=src python -m project_cnn_mammo.cli --config configs/sprints.toml smoke

lint:
	python -m ruff check src tests || true

scan:
	python scripts/check_secrets.py .

gate:
	python scripts/quality_gate.py outputs/mlruns outputs/metrics.json

clean:
	rm -rf outputs mlruns artifacts .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
