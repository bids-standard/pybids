#!/bin/bash

echo Installing dependencies

source tools/ci/activate.sh
source tools/ci/env.sh

set -eu

# Required variables
echo EXTRA_PIP_FLAGS = $EXTRA_PIP_FLAGS
echo CHECK_TYPE = $CHECK_TYPE

set -x

if [ -n "$EXTRA_PIP_FLAGS" ]; then
    EXTRA_PIP_FLAGS=${!EXTRA_PIP_FLAGS}
fi

pip install $EXTRA_PIP_FLAGS "pybids[$CHECK_TYPE]"

set +eux

echo Done installing dependencies
