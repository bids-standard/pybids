from os.path import join

import pytest

from bids.layout import BIDSLayout


# Fixture uses in the rest of the tests
@pytest.fixture(scope="session")
def layout_7t_trt(tests_dir):
    return BIDSLayout(tests_dir / 'data' / '7t_trt')


@pytest.fixture(scope="session")
def layout_7t_trt_relpath(tests_dir):
    return BIDSLayout(tests_dir / 'data' / '7t_trt', absolute_paths=False)


@pytest.fixture(scope="session")
def layout_ds005(tests_dir):
    return BIDSLayout(tests_dir / 'data' / 'ds005')


@pytest.fixture(scope="session")
def layout_ds005_no_validate(tests_dir):
    return BIDSLayout(tests_dir / 'data' / 'ds005', validate=False)


@pytest.fixture(scope="session")
def layout_ds117(tests_dir):
    return BIDSLayout(tests_dir / 'data' / 'ds000117')


@pytest.fixture(scope="session")
def layout_ds005_derivs(tests_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    layout = BIDSLayout(data_dir)
    deriv_dir = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives(deriv_dir)
    return layout


@pytest.fixture(scope="session")
def layout_ds005_deriv_dummy_vxxx(tests_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    layout = BIDSLayout(data_dir)
    deriv_dir = tests_dir / 'data' / 'ds005_derivs' / 'dummy-vx.x.x'
    layout.add_derivatives(deriv_dir)
    return layout


@pytest.fixture(scope="session")
def layout_ds005_deriv_both_dummies(tests_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    layout = BIDSLayout(data_dir)
    deriv_dir1 = tests_dir / 'data' / 'ds005_derivs' / 'dummy-vx.x.x'
    deriv_dir2 = tests_dir / 'data' / 'ds005_derivs' / 'dummy'
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(scope="session")
def db_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return fn


@pytest.fixture(scope="session",
                params=[None, "bidsdb", "bidsdb"])
def layout_ds005_multi_derivs(tests_dir, request, db_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    database_path = str(db_dir / request.param) if request.param else None

    layout = BIDSLayout(data_dir,
                        database_path=database_path)
    deriv_dir1 = tests_dir / 'data' / 'ds005_derivs' / 'dummy'
    deriv_dir2 = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout

@pytest.fixture(scope="session")
def layout_ds005_deriv_name_collision(tests_dir, request):
    data_dir = tests_dir / 'data' / 'ds005'

    layout = BIDSLayout(data_dir)
    deriv_dir1 = tests_dir / 'data' / 'ds005_derivs' / 'dummy'
    deriv_dir2 = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(
    scope="session", params=[None, "bidsdb-synth", "bidsdb-synth"])
def layout_synthetic(tests_dir, request, db_dir):
    path = tests_dir / 'data' / 'synthetic'
    database_path = str(db_dir / request.param) if request.param else None
    return BIDSLayout(path, derivatives=True,
                      database_path=database_path)

@pytest.fixture(scope="session")
def layout_synthetic_nodb(tests_dir, request, db_dir):
    path = tests_dir / 'data' / 'synthetic'
    return BIDSLayout(path, derivatives=True)
