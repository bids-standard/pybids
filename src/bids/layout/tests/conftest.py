import shutil
from os.path import join
from functools import lru_cache

import pytest
import requests

from bids.layout import BIDSLayout


# Fixture uses in the rest of the tests
@pytest.fixture(scope='session')
def layout_7t_trt(tests_dir):
    return BIDSLayout(tests_dir / 'data' / '7t_trt')


@pytest.fixture(scope='session')
def layout_ds005(tests_dir):
    return BIDSLayout(tests_dir / 'data' / 'ds005')


@pytest.fixture(scope='session')
def layout_ds005_no_validate(tests_dir):
    return BIDSLayout(tests_dir / 'data' / 'ds005', validate=False)


@pytest.fixture(scope='session')
def layout_ds117(tests_dir):
    return BIDSLayout(tests_dir / 'data' / 'ds000117')


@pytest.fixture(scope='session')
def layout_ds005_derivs(tests_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    layout = BIDSLayout(data_dir)
    deriv_dir = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives(deriv_dir)
    return layout


@pytest.fixture(scope='session')
def layout_ds005_deriv_dummy_vxxx(tests_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    layout = BIDSLayout(data_dir)
    deriv_dir = tests_dir / 'data' / 'ds005_derivs' / 'dummy-vx.x.x'
    layout.add_derivatives(deriv_dir)
    return layout


@pytest.fixture(scope='session')
def layout_ds005_deriv_both_dummies(tests_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    layout = BIDSLayout(data_dir)
    deriv_dir1 = tests_dir / 'data' / 'ds005_derivs' / 'dummy-vx.x.x'
    deriv_dir2 = tests_dir / 'data' / 'ds005_derivs' / 'dummy'
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(scope='session')
def db_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('data')
    return fn


@pytest.fixture(scope='session', params=[None, 'bidsdb', 'bidsdb'])
def layout_ds005_multi_derivs(tests_dir, request, db_dir):
    data_dir = tests_dir / 'data' / 'ds005'
    database_path = str(db_dir / request.param) if request.param else None

    layout = BIDSLayout(data_dir, database_path=database_path)
    deriv_dir1 = tests_dir / 'data' / 'ds005_derivs' / 'dummy'
    deriv_dir2 = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(scope='session')
def layout_ds005_deriv_name_collision(tests_dir, request):
    data_dir = tests_dir / 'data' / 'ds005'

    layout = BIDSLayout(data_dir)
    deriv_dir1 = tests_dir / 'data' / 'ds005_derivs' / 'dummy'
    deriv_dir2 = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(scope='session', params=[None, 'bidsdb-synth', 'bidsdb-synth'])
def layout_synthetic(tests_dir, request, db_dir):
    path = tests_dir / 'data' / 'synthetic'
    database_path = str(db_dir / request.param) if request.param else None
    return BIDSLayout(path, derivatives=True, database_path=database_path)


@pytest.fixture(scope='session')
def layout_synthetic_nodb(tests_dir, request, db_dir):
    path = tests_dir / 'data' / 'synthetic'
    return BIDSLayout(path, derivatives=True)


@pytest.fixture
def temporary_dataset(tmp_path, tests_dir):
    path = tests_dir / 'data' / 'ds005'
    shutil.copytree(path, tmp_path / 'ds005')
    return tmp_path / 'ds005'


@lru_cache(maxsize=1)
def _schema_hosts_reachable(timeout=3):
    hosts = [
        "https://bids.neuroimaging.io",
        "https://bids-specification.readthedocs.io/en/latest/schema.json",
    ]
    for host in hosts:
        try:
            response = requests.head(host, timeout=timeout, allow_redirects=True)
            if response.status_code >= 400:
                return False
        except requests.RequestException:
            return False
    return True


@pytest.fixture(scope='session')
def require_schema_network():
    if not _schema_hosts_reachable():
        pytest.skip(
            "Skipping schema retrieval tests because bids.neuroimaging.io or schema host is unreachable."
        )


_SCHEMA_NETWORK_MODULES = {
    "bids.layout.tests.test_schema_config",
    "bids.layout.tests.test_schema_pattern_validation",
    "bids.layout.tests.test_schema_version_differences",
    "bids.layout.tests.test_schema_vs_json_config",
}


@pytest.fixture(autouse=True)
def _skip_schema_network_tests_when_unreachable(request):
    module_name = getattr(request.module, "__name__", "")
    if module_name in _SCHEMA_NETWORK_MODULES and not _schema_hosts_reachable():
        pytest.skip(
            "Skipping schema retrieval tests because bids.neuroimaging.io or schema host is unreachable."
        )
