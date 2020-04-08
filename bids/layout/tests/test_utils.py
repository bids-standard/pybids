import pytest
from ..utils import BIDSMetadata


def test_bidsmetadata_class():
    md = BIDSMetadata("fakefile")
    with pytest.raises(KeyError) as err:
        md["Missing"]
    assert "Metadata term 'Missing' unavailable for file fakefile." in str(err)
    md["Missing"] = 1
    assert md["Missing"] == 1
