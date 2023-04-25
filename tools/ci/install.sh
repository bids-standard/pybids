#!/bin/bash

echo Installing pybids

source tools/ci/activate.sh
source tools/ci/env.sh

set -eu

# Required variables
echo INSTALL_TYPE = $INSTALL_TYPE
echo CHECK_TYPE = $CHECK_TYPE
echo EXTRA_PIP_FLAGS = $EXTRA_PIP_FLAGS
echo REQUIREMENTS = $REQUIREMENTS

set -x

if [ -n "$EXTRA_PIP_FLAGS" ]; then
    EXTRA_PIP_FLAGS=${!EXTRA_PIP_FLAGS}
fi

pip install $EXTRA_PIP_FLAGS $REQUIREMENTS $ARCHIVE

# Basic import check
python -c 'import bids; print(bids.__version__)'

set +eux

echo Done installing pybids
