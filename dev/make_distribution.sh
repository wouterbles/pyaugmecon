#!/bin/bash

set -o pipefail
set -e

FWDIR="$(cd "`dirname "${BASH_SOURCE[0]}"`"; pwd)"
cd "$FWDIR/.."

# Clean up previous build artifacts
rm -rf dist
rm -rf pyaugmecon.egg-info

# Build source and wheel distribution using the `build` package
python -m build

# Upload the distribution files to PyPI using twine
twine upload dist/*
