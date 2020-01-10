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


@pytest.fixture(scope="session")
def db_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return fn


@pytest.fixture(scope="module",
                params=[None, "bidsdb", "bidsdb"])
def layout_ds005_multi_derivs(request, db_dir):
    data_dir = join(get_test_data_path(), 'ds005')
    database_path = str(db_dir / request.param) if request.param else None

    layout = BIDSLayout(data_dir,
                        database_path=database_path)
    deriv_dir1 = join(get_test_data_path(), 'ds005_derivs')
    deriv_dir2 = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(
    scope="module", params=[None, "bidsdb-synth", "bidsdb-synth"])
def layout_synthetic(request, db_dir):
    path = join(get_test_data_path(), 'synthetic')
    database_path = str(db_dir / request.param) if request.param else None
    return BIDSLayout(path, derivatives=True,
                      database_path=database_path)

@pytest.fixture(scope="module")
def layout_synthetic_nodb(request, db_dir):
    path = join(get_test_data_path(), 'synthetic')
    return BIDSLayout(path, derivatives=True)
