#!/bin/bash

# Check if package name is supplied
if [ -z "$1" ]; then
  echo "Usage: $0 <package_name>"
  exit 1
fi

PACKAGE_NAME=$1

# Download the specific wheel file built for Python 3.11
pip download $PACKAGE_NAME --python-version 3.11 --only-binary=:all:

# Extract the version of the downloaded package
PACKAGE_VERSION=$(ls ${PACKAGE_NAME}-*cp311*.whl | grep -oP '(?<=-)[0-9]+(\.[0-9]+)*')

# Rename the wheel file to make it compatible with Python 3.12
mv ${PACKAGE_NAME}-*cp311*.whl ${PACKAGE_NAME}-${PACKAGE_VERSION}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Install the renamed wheel file
pip install ${PACKAGE_NAME}-${PACKAGE_VERSION}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

