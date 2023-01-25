SCRATCH_DIR=testing/scratch
NEW_SCRATCH_DIR=testing/scratcher

null:
	@:

clean:
	rm -rf build
	rm -rf pycinema.egg-info
	rm -rf dist
	rm -rf $(SCRATCH_DIR)

example:
	@rm -rf $(SCRATCH_DIR)
	@if [ ! -d "$(SCRATCH_DIR)" ]; then\
		echo "Creating scratch dir";\
		mkdir $(SCRATCH_DIR);\
	fi
	@echo "Creating test area $(SCRATCH_DIR)"
	@cp -rf pycinema $(SCRATCH_DIR)/pycinema
	@cp -rf data $(SCRATCH_DIR)/data
	@cp -rf examples $(SCRATCH_DIR)/examples
	@cp -rf fonts $(SCRATCH_DIR)/fonts

module:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	python3 setup.py sdist

module-upload:
	twine upload dist/*
