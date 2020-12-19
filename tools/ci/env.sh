SETUP_REQUIRES="pip setuptools>=30.3.0 wheel"

# Numpy and scipy upload nightly/weekly/intermittent wheels
NIGHTLY_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
STAGING_WHEELS="https://pypi.anaconda.org/multibuild-wheels-staging/simple"
PRE_PIP_FLAGS="--pre --extra-index-url $NIGHTLY_WHEELS --extra-index-url $STAGING_WHEELS"

# XXX This is necessary to deal with scipy version metadata mismatches
# We should aim to remove this ASAP.
# Last check: 2020.12.19
PRE_PIP_FLAGS="--use-deprecated=legacy-resolver $PRE_PIP_FLAGS"
