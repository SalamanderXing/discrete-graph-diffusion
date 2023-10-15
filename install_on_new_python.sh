#!/bin/bash

# Check if package name is supplied
if [ -z "$1" ]; then
  echo "Usage: $0 <package_name> [pip_options]"
  exit 1
fi

PACKAGE_NAME=$1
shift  # Remove the first argument (package name) to keep only additional pip options
PIP_OPTIONS="$@"

# Download the specific wheel file built for Python 3.11 to /tmp
python -m pip download $PACKAGE_NAME $PIP_OPTIONS --python-version 3.11 --only-binary=:all: --dest /tmp

# Extract the version of the downloaded package
PACKAGE_VERSION=$(ls /tmp/${PACKAGE_NAME}-*cp311*.whl | grep -oP '(?<=-)[0-9]+(\.[0-9]+)*')

# Rename the wheel file to make it compatible with Python 3.12
mv /tmp/${PACKAGE_NAME}-*cp311*.whl /tmp/${PACKAGE_NAME}-${PACKAGE_VERSION}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Install the renamed wheel file
python -m pip install /tmp/${PACKAGE_NAME}-${PACKAGE_VERSION}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl $PIP_OPTIONS

