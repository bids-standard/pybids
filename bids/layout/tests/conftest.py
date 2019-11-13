from os.path import join

import pytest
import tempfile
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


# Fixture uses in the rest of the tests
@pytest.fixture(scope="module")
def layout_7t_trt():
    data_dir = join(get_test_data_path(), '7t_trt')
    return BIDSLayout(data_dir)


@pytest.fixture(scope="module")
def layout_7t_trt_relpath():
    data_dir = join(get_test_data_path(), '7t_trt')
    return BIDSLayout(data_dir, absolute_paths=False)


@pytest.fixture(scope="module")
def layout_ds005():
    data_dir = join(get_test_data_path(), 'ds005')
    return BIDSLayout(data_dir)


@pytest.fixture(scope="module")
def layout_ds117():
    data_dir = join(get_test_data_path(), 'ds000117')
    return BIDSLayout(data_dir)


@pytest.fixture(scope="module")
def layout_ds005_derivs():
    data_dir = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(data_dir)
    deriv_dir = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives(deriv_dir)
    return layout


fdir = tempfile.mkdtemp()

@pytest.fixture(scope="module",
                params=[None, "bidsdb", "bidsdb"])
def layout_ds005_multi_derivs(request):
    data_dir = join(get_test_data_path(), 'ds005')
    database_dir = join(fdir, request.param) if request.param else None

    layout = BIDSLayout(data_dir,
                        database_dir=database_dir)
    deriv_dir1 = join(get_test_data_path(), 'ds005_derivs')
    deriv_dir2 = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(
    scope="module", params=[None, "bidsdb-synth", "bidsdb-synth"])
def layout_synthetic(request):
    path = join(get_test_data_path(), 'synthetic')
    database_dir = join(fdir, request.param) if request.param else None
    return BIDSLayout(path, derivatives=True,
                      database_dir=database_dir)
