.PHONY: setup reset remove clean run

VENV_PYTHON = .venv/bin/python

pipeline:
	@echo "...Running all steps..."
	@PYTHONPATH=code $(VENV_PYTHON) code/controller_all.py

run_FL:
	@echo "...Running FL Template..."
	@PYTHONPATH=code $(VENV_PYTHON) code/main_4_FL_Template.py

setup:
	@echo "...Creating project folders (if missing)..."
	@PYTHONPATH=code $(VENV_PYTHON) code/utils/setup_folders.py

reset:
	@echo "...Resetting project folders..."
	@PYTHONPATH=code $(VENV_PYTHON) code/utils/setup_folders.py --overwrite

remove:
	@echo "...Removing project folders..."
	@PYTHONPATH=code $(VENV_PYTHON) code/utils/setup_folders.py --remove-only

clean:
	@echo "...Cleaning up __pycache__..."
	@find . -type d -name "__pycache__" -exec rm -r {} +

