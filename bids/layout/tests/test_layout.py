""" Tests of functionality in the layout module--mostly related to the
BIDSLayout class."""

import os
import re
from os.path import join, abspath, basename
from pathlib import Path
import shutil
import json

import numpy as np
import pytest

from bids.layout import BIDSLayout, Query
from bids.layout.models import Config
from bids.layout.index import BIDSLayoutIndexer, _check_path_matches_patterns, _regexfy
from bids.layout.utils import PaddedInt
from bids.tests import get_test_data_path
from bids.utils import natural_sort

from bids.exceptions import (
    BIDSChildDatasetError,
    BIDSDerivativesValidationError,
    BIDSValidationError,
    NoMatchError,
    TargetError,
)


def test_layout_init(layout_7t_trt):
    assert isinstance(layout_7t_trt.files, dict)


@pytest.mark.parametrize(
    'index_metadata,query,result',
    [
        (True, {}, 3.0),
        (False, {}, None),
        (True, {}, 3.0),
        (True, {'task': 'rest'}, 3.0),
        (True, {'task': 'rest', 'extension': ['.nii.gz']}, 3.0),
        (True, {'task': 'rest', 'extension': '.nii.gz'}, 3.0),
        (True, {'task': 'rest', 'extension': ['.nii.gz', '.json'], 'return_type': 'file'}, 3.0),
    ])
def test_index_metadata(index_metadata, query, result, mock_config):
    data_dir = join(get_test_data_path(), '7t_trt')
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(index_metadata=index_metadata),
        **query
    )

    sample_file = layout.get(task='rest', extension='.nii.gz',
                             acquisition='fullbrain')
    assert bool(sample_file)
    metadata = sample_file[0].get_metadata()
    assert metadata.get('RepetitionTime') == result



@pytest.mark.parametrize('config_type', [str, Path])
def test_config_filename(config_type):
    data_path = Path(get_test_data_path())
    # Use custom config that replaces session with oligarchy
    config_path = data_path.parent / 'bids_specs_with_oligarchy.json'
    layout = BIDSLayout(data_path / "7t_trt", config=config_type(config_path))
    # Validate that we are using the desired configuration
    assert 'oligarchy' in layout.get_entities()


def test_layout_repr(layout_7t_trt):
    assert "Subjects: 10 | Sessions: 20 | Runs: 20" in str(layout_7t_trt)


def test_invalid_dataset_description(tmp_path):
    shutil.copytree(join(get_test_data_path(), '7t_trt'), tmp_path / "7t_dset")
    (tmp_path / "7t_dset" / "dataset_description.json").write_text(
        "I am not a valid json file"
    )
    with pytest.raises(BIDSValidationError) as exc:
        BIDSLayout(tmp_path / "7t_dset")

    assert "is not a valid json file" in str(exc.value)


def test_layout_repr_overshadow_run(tmp_path):
    """A test creating a layout to replicate #681."""
    shutil.copytree(join(get_test_data_path(), '7t_trt'), tmp_path / "7t_trt")
    (tmp_path / "7t_trt" / "sub-01" / "ses-1" / "sub-01_ses-1_scans.json").write_text(
        json.dumps({"run": {"Description": "metadata to cause #681"}})
    )
    assert "Subjects: 10 | Sessions: 20 | Runs: 20" in str(BIDSLayout(tmp_path / "7t_trt"))

# def test_layout_copy(layout_7t_trt):
#     # Largely a smoke test to guarantee that copy() does not blow
#     # see https://github.com/bids-standard/pybids/pull/400#issuecomment-467961124
#     import copy
#     l = layout_7t_trt

#     lcopy = copy.copy(l)
#     assert repr(lcopy) == repr(l)
#     assert str(lcopy) == str(l)

#     lcopy = copy.deepcopy(l)
#     assert repr(lcopy) == repr(l)
#     assert str(lcopy) == str(l)


def test_load_description(layout_7t_trt):
    # Should not raise an error
    assert hasattr(layout_7t_trt, 'description')
    assert layout_7t_trt.description['Name'] == '7t_trt'
    assert layout_7t_trt.description['BIDSVersion'] == "1.0.0rc3"


def test_get_file(layout_ds005_derivs):
    layout = layout_ds005_derivs

    # relative path in BIDS-Raw
    orig_file = 'sub-13/func/sub-13_task-mixedgamblestask_run-01_bold.nii.gz'
    target = os.path.join(*orig_file.split('/'))
    assert layout.get_file(target)
    assert layout.get_file(target, scope='raw')
    assert not layout.get_file(target, scope='derivatives')

    # absolute path in BIDS-Raw
    target = (layout.root + '/' + orig_file).split('/')
    target = os.path.sep + os.path.join(*target)
    assert layout.get_file(target)
    assert layout.get_file(target, scope='raw')
    assert not layout.get_file(target, scope='derivatives')

    # relative path in derivatives pipeline
    orig_file = 'derivatives/events/sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv'
    target = os.path.join(*orig_file.split('/'))
    assert layout.get_file(target)
    assert not layout.get_file(target, scope='raw')
    assert layout.get_file(target, scope='derivatives')

    # absolute path in derivatives pipeline
    target = (layout.root + '/' + orig_file).split('/')
    target = os.path.sep + os.path.join(*target)
    assert layout.get_file(target)
    assert not layout.get_file(target, scope='raw')
    assert layout.get_file(target, scope='derivatives')
    assert layout.get_file(target, scope='events')

    # No such file
    assert not layout.get_file('bleargh')
    assert not layout.get_file('/absolute/bleargh')


class TestDerivativeAsRoot:
    def test_dataset_without_datasettype_parsed_as_raw(self):
        dataset_path = Path("ds005_derivs", "format_errs", "no_dataset_type")
        unvalidated = BIDSLayout(
            Path(get_test_data_path())/dataset_path,
            validate=False
        )
        assert len(unvalidated.get()) == 4
        with pytest.raises(ValueError):
            unvalidated.get(desc="preproc")

        validated = BIDSLayout(Path(get_test_data_path())/dataset_path)
        assert len(validated.get()) == 1

    def test_derivative_indexing_forced_with_is_derivative(self):
        dataset_path = Path("ds005_derivs", "format_errs", "no_type_or_description")
        unvalidated = BIDSLayout(
            Path(get_test_data_path())/dataset_path,
            is_derivative=True,
            validate=False
        )
        assert len(unvalidated.get()) == 4
        assert len(unvalidated.get(desc="preproc")) == 3

    def test_forced_derivative_indexing_fails_validation(self):
        dataset_path = Path("ds005_derivs", "format_errs", "no_type_or_description")
        with pytest.raises(BIDSDerivativesValidationError):
            BIDSLayout(
                Path(get_test_data_path())/dataset_path,
                is_derivative=True,
                validate=True
            )

    def test_dataset_missing_generatedby_fails_validation(self):
        dataset_path = Path("ds005_derivs", "format_errs", "no_pipeline_description")
        with pytest.raises(BIDSDerivativesValidationError):
            BIDSLayout(Path(get_test_data_path())/dataset_path)


    def test_correctly_formatted_derivative_loads_as_derivative(self):
        dataset_path = Path("ds005_derivs", "dummy")
        layout = BIDSLayout(Path(get_test_data_path())/dataset_path, validate=False)
        assert len(layout.get()) == 4
        assert len(layout.get(desc="preproc")) == 3

    @pytest.mark.parametrize(
        "dataset_path",
        [
            Path("ds005_derivs", "dummy"),
            Path("ds005_derivs", "format_errs", "no_pipeline_description")
        ]
    )
    def test_derivative_datasets_load_with_no_validation(self, dataset_path):
        layout = BIDSLayout(
            Path(get_test_data_path())/dataset_path,
            validate=False
        )
        assert len(layout.get()) == 4
        assert len(layout.get(desc="preproc")) == 3

def test_get_metadata(layout_7t_trt):
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    target = target.split('/')
    result = layout_7t_trt.get_metadata(join(layout_7t_trt.root, *target))
    assert result['RepetitionTime'] == 3.0


def test_get_metadata2(layout_7t_trt):
    target = 'sub-03/ses-1/fmap/sub-03_ses-1_run-1_phasediff.nii.gz'
    target = target.split('/')
    result = layout_7t_trt.get_metadata(join(layout_7t_trt.root, *target))
    assert result['EchoTime1'] == 0.006


def test_get_metadata3(layout_7t_trt):
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    target = target.split('/')
    result = layout_7t_trt.get_metadata(join(layout_7t_trt.root, *target))
    assert result['EchoTime'] == 0.020

    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    target = target.split('/')
    result = layout_7t_trt.get_metadata(join(layout_7t_trt.root, *target))
    assert result['EchoTime'] == 0.017


def test_get_metadata4(layout_ds005):
    target = 'sub-03/anat/sub-03_T1w.nii.gz'
    target = target.split('/')
    result = layout_ds005.get_metadata(join(layout_ds005.root, *target))
    assert result == {}


def test_get_metadata_meg(layout_ds117):
    funcs = ['get_subjects', 'get_sessions', 'get_tasks', 'get_runs',
             'get_acquisitions', 'get_procs']
    assert all([hasattr(layout_ds117, f) for f in funcs])
    procs = layout_ds117.get_procs()
    assert procs == ['sss']
    target = 'sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-01_meg.fif'
    target = target.split('/')
    result = layout_ds117.get_metadata(join(layout_ds117.root, *target))
    metadata_keys = ['MEGChannelCount', 'SoftwareFilters', 'SubjectArtefactDescription']
    assert all([k in result for k in metadata_keys])


def test_get_metadata5(layout_7t_trt):
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    target = target.split('/')
    result = layout_7t_trt.get_metadata(
        join(layout_7t_trt.root, *target), include_entities=True)
    assert result['EchoTime'] == 0.020
    assert result['subject'] == '01'
    assert result['acquisition'] == 'fullbrain'


def test_get_metadata_via_bidsfile(layout_7t_trt):
    ''' Same as test_get_metadata5, but called through BIDSFile. '''
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    target = target.split('/')
    path = join(layout_7t_trt.root, *target)
    result = layout_7t_trt.files[path].get_metadata()
    assert result['EchoTime'] == 0.020
    # include_entities is False when called through a BIDSFile
    assert 'subject' not in result


def test_get_metadata_error(layout_7t_trt):
    ''' Same as test_get_metadata5, but called through BIDSFile. '''
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    target = target.split('/')
    path = join(layout_7t_trt.root, *target)
    result = layout_7t_trt.files[path].get_metadata()
    with pytest.raises(KeyError) as err:
        result['Missing']
    assert "Metadata term 'Missing' unavailable for file {}".format(path) in str(err.value)

    result = layout_7t_trt.get_metadata(path)
    with pytest.raises(KeyError) as err:
        result['Missing']
    assert "Metadata term 'Missing' unavailable for file {}".format(path) in str(err.value)


def test_get_with_bad_target(layout_7t_trt):
    with pytest.raises(TargetError) as exc:
        layout_7t_trt.get(target='unicorn')
    msg = str(exc.value)
    assert 'subject' in msg and 'reconstruction' in msg and 'proc' in msg
    with pytest.raises(TargetError) as exc:
        layout_7t_trt.get(target='sub')
    msg = str(exc.value)
    assert 'subject' in msg and 'reconstruction' not in msg


def test_get_bvals_bvecs(layout_ds005):
    dwifile = layout_ds005.get(subject="01", datatype="dwi")[0]
    result = layout_ds005.get_bval(dwifile.path)
    assert result == abspath(join(layout_ds005.root, 'dwi.bval'))

    result = layout_ds005.get_bvec(dwifile.path)
    assert result == abspath(join(layout_ds005.root, 'dwi.bvec'))


def test_get_subjects(layout_7t_trt):
    result = layout_7t_trt.get_subjects()
    predicted = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    assert set(predicted) == set(result)


def test_get_fieldmap(layout_7t_trt):
    target = 'sub-03/ses-1/func/sub-03_ses-1_task-' \
             'rest_acq-fullbrain_run-1_bold.nii.gz'
    target = target.split('/')
    result = layout_7t_trt.get_fieldmap(join(layout_7t_trt.root, *target))
    assert result["suffix"] == "phasediff"
    assert result["phasediff"].endswith('sub-03_ses-1_run-1_phasediff.nii.gz')


def test_get_fieldmap2(layout_7t_trt):
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    target = target.split('/')
    result = layout_7t_trt.get_fieldmap(join(layout_7t_trt.root, *target))
    assert result["suffix"] == "phasediff"
    assert result["phasediff"].endswith('sub-03_ses-2_run-2_phasediff.nii.gz')


def test_bids_json(layout_7t_trt):
    res = layout_7t_trt.get(return_type='id', target='run')
    assert set(res) == {1, 2}
    res = layout_7t_trt.get(return_type='id', target='session')
    assert set(res) == {'1', '2'}


def test_get_return_type_dir(layout_7t_trt, layout_7t_trt_relpath):
    query = dict(target='subject', return_type='dir')
    # In case of relative paths
    res_relpath = layout_7t_trt_relpath.get(**query)
    # returned directories should be in sorted order so we can match exactly
    target_relpath = ["sub-{:02d}".format(i) for i in range(1, 11)]
    assert target_relpath == res_relpath

    res = layout_7t_trt.get(**query)
    target = [
        os.path.join(get_test_data_path(), '7t_trt', p)
        for p in target_relpath
    ]
    assert target == res

    # and we can overload the value for absolute_path in .get call
    res_relpath2 = layout_7t_trt.get(absolute_paths=False, **query)
    assert target_relpath == res_relpath2
    res2 = layout_7t_trt_relpath.get(absolute_paths=True, **query)
    assert target == res2


@pytest.mark.parametrize("acq", [None, Query.NONE])
def test_get_val_none(layout_7t_trt, acq):
    t1w_files = layout_7t_trt.get(subject='01', session='1', suffix='T1w')
    assert len(t1w_files) == 1
    assert 'acq' not in t1w_files[0].path
    t1w_files = layout_7t_trt.get(
        subject='01', session='1', suffix='T1w', acquisition=acq)
    assert len(t1w_files) == 1
    bold_files = layout_7t_trt.get(
        subject='01', session='1', suffix='bold', acquisition=acq)
    assert len(bold_files) == 0


def test_get_val_enum_any(layout_7t_trt):
    t1w_files = layout_7t_trt.get(
        subject='01', session='1', suffix='T1w', acquisition=Query.ANY,
        extension=Query.ANY)
    assert not t1w_files
    bold_files = layout_7t_trt.get(subject='01', session='1', run=1,
                                  suffix='bold', acquisition=Query.ANY)
    assert len(bold_files) == 2


def test_get_val_enum_any_optional(layout_7t_trt, layout_ds005):
    # layout with sessions
    query = {
        "subject": "01",
        "run": 1,
        "suffix": "bold",
    }
    bold_files = layout_7t_trt.get(session=Query.OPTIONAL, **query)
    assert len(bold_files) == 3

    # layout without sessions
    bold_files = layout_ds005.get(session=Query.REQUIRED, **query)
    assert not bold_files
    bold_files = layout_ds005.get(session=Query.OPTIONAL, **query)
    assert len(bold_files) == 1


def test_get_return_sorted(layout_7t_trt):
    bids_files = layout_7t_trt.get(target='subject')
    paths = [r.path for r in bids_files]
    assert natural_sort(paths) == paths

    files = layout_7t_trt.get(target='subject', return_type='file')
    assert files == paths


def test_ignore_files(layout_ds005):
    data_dir = join(get_test_data_path(), 'ds005')
    target1 = join(data_dir, 'models', 'ds-005_type-test_model.json')
    target2 = join(data_dir, 'models', 'extras', 'ds-005_type-test_model.json')
    layout1 = BIDSLayout(data_dir, validate=False)
    assert target1 not in layout_ds005.files
    assert target1 not in layout1.files
    assert target2 not in layout1.files
    # now the models/ dir should show up, because passing ignore explicitly
    # overrides the default - but 'model/extras/' should still be ignored
    # because of the regex.
    ignore = [re.compile('xtra'), 'dummy']
    indexer = BIDSLayoutIndexer(validate=False, ignore=ignore)
    layout2 = BIDSLayout(data_dir, indexer=indexer)
    assert target1 in layout2.files
    assert target2 not in layout2.files


def test_force_index(layout_ds005):
    data_dir = join(get_test_data_path(), 'ds005')
    target = join(data_dir, 'models', 'ds-005_type-test_model.json')
    indexer = BIDSLayoutIndexer(force_index=[re.compile('models(/.*)?')])
    model_layout = BIDSLayout(data_dir, validate=True, indexer=indexer)
    assert target not in layout_ds005.files
    assert target in model_layout.files
    assert 'all' not in model_layout.get_subjects()
    for f in model_layout.files.values():
        assert 'derivatives' not in f.path


def test_nested_include_exclude():
    data_dir = join(get_test_data_path(), 'ds005')
    target1 = join(data_dir, 'models', 'ds-005_type-test_model.json')
    target2 = join(data_dir, 'models', 'extras', 'ds-005_type-test_model.json')

    # Excluding a directory will disallow indexing further, even if forced
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            force_index=[os.path.join('models', 'extras')],
            ignore=['models'],
        ),
    )
    assert not layout.get_file(target1)
    assert not layout.get_file(target2)

    # To nest patterns, we must allow searching within
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            force_index=[os.path.join('models', 'extras')],
            ignore=[re.compile('^/models/.+$')],
        ),
    )
    assert not layout.get_file(target1)
    assert layout.get_file(target2)

    # Including a directory will disallow marking ignore subdirectories within
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            force_index=['models'],
            ignore=[os.path.join('models', 'extras')],
        ),
    )
    assert layout.get_file(target1)
    assert layout.get_file(target2)

    # To nest a directory exclusion within an inclusion, use regex
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            force_index=[re.compile('^/models/?$')],
            ignore=[os.path.join('models', 'extras')],
        ),
    )
    assert layout.get_file(target1)
    assert not layout.get_file(target2)

    # Force file inclusion despite directory-level exclusion
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            force_index=['models', target2],
            ignore=[os.path.join('models', 'extras')],
        ),
    )
    assert layout.get_file(target1)
    assert layout.get_file(target2)

    # To nest patterns, we must allow searching within
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            force_index=[os.path.join('models', 'extras')],
            ignore=[re.compile('^/models/.+$')],
        ),
    )
    assert not layout.get_file(target1)
    assert layout.get_file(target2)

    # With the validator, these paths will not be valid anyways
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=True,
            force_index=['models'],
            ignore=[os.path.join('models', 'extras')],
        ),
    )
    assert layout.get_file(target1)
    assert layout.get_file(target2)

    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=True,
            force_index=[re.compile('^/models/?$')],
            ignore=[os.path.join('models', 'extras')],
        ),
    )
    assert layout.get_file(target1)
    assert not layout.get_file(target2)

    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=True,
            force_index=['models', target2],
            ignore=[os.path.join('models', 'extras')],
        ),
    )
    assert layout.get_file(target1)
    assert layout.get_file(target2)


def test_nested_include_exclude_with_regex():
    # ~same as above test, but use regexps instead of strings
    patt1 = re.compile(r'.*dels/?$')
    patt2 = re.compile(r'.*xtra.*')
    data_dir = join(get_test_data_path(), 'ds005')
    target1 = join(data_dir, 'models', 'ds-005_type-test_model.json')
    target2 = join(data_dir, 'models', 'extras', 'ds-005_type-test_model.json')

    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            ignore=[patt2],
            force_index=[patt1],
        )
    )
    assert layout.get_file(target1)
    assert not layout.get_file(target2)

    # If ignore matches a folder, it won't be indexed
    patt1 = re.compile(r'.*dels/.+$')
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=False,
            ignore=[patt1],
            force_index=[patt2],
        )
    )
    assert not layout.get_file(target1)
    assert layout.get_file(target2)

    # With valid_only no path should be indexed under models/
    patt1 = re.compile(r'.*dels/ds-005.*')
    patt2 = re.compile(r'.*xtra.*')
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=True,
            ignore=[patt2],
            force_index=[patt1],
        )
    )
    assert layout.get_file(target1)
    assert not layout.get_file(target2)

    # If ignore matches a folder, it won't be indexed
    patt1 = re.compile(r'.*dels/.+$')
    layout = BIDSLayout(
        data_dir,
        indexer=BIDSLayoutIndexer(
            validate=True,
            ignore=[patt1],
            force_index=[patt2],
        )
    )
    assert not layout.get_file(target1)
    assert layout.get_file(target2)


def test_layout_with_derivs(layout_ds005_derivs):
    assert layout_ds005_derivs.root == join(get_test_data_path(), 'ds005')
    assert isinstance(layout_ds005_derivs.files, dict)
    assert len(layout_ds005_derivs.derivatives) == 1
    deriv = layout_ds005_derivs.derivatives['events']
    assert deriv.files
    assert len(deriv.files) == 2
    event_file = "sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv"
    deriv_files = [basename(f) for f in list(deriv.files.keys())]
    assert event_file in deriv_files
    assert 'roi' in deriv.entities
    assert 'subject' in deriv.entities


def test_accessing_deriv_by_pipeline_name_is_deprecated(layout_ds005_deriv_dummy_vxxx):
    with pytest.deprecated_call():
        deriv = layout_ds005_deriv_dummy_vxxx.derivatives['dummy']
    assert deriv.files
    assert len(deriv.files) == 4


def test_cant_access_nonexistant_deriv_by_key(layout_ds005_deriv_dummy_vxxx):
    with pytest.raises(KeyError):
        layout_ds005_deriv_dummy_vxxx.derivatives['foo']


def test_accessing_deriv_by_pipeline_name_via_method(layout_ds005_deriv_dummy_vxxx):
    deriv = layout_ds005_deriv_dummy_vxxx.derivatives.get_pipeline('dummy')
    assert deriv.files
    assert len(deriv.files) == 4


def test_cant_get_nonexistant_deriv_via_method(layout_ds005_deriv_dummy_vxxx):
    with pytest.raises(KeyError):
        layout_ds005_deriv_dummy_vxxx.derivatives.get_pipeline('foo')


def test_cant_get_deriv_with_duplicate_pipeline_via_method(layout_ds005_deriv_both_dummies):
    with pytest.raises(BIDSChildDatasetError):
        layout_ds005_deriv_both_dummies.derivatives.get_pipeline('dummy')


def test_layout_with_multi_derivs(layout_ds005_multi_derivs):
    assert layout_ds005_multi_derivs.root == join(get_test_data_path(), 'ds005')
    assert isinstance(layout_ds005_multi_derivs.files, dict)
    assert len(layout_ds005_multi_derivs.derivatives) == 2
    deriv = layout_ds005_multi_derivs.derivatives['events']
    assert deriv.files
    assert len(deriv.files) == 2
    deriv = layout_ds005_multi_derivs.derivatives['dummy']
    assert deriv.files
    assert len(deriv.files) == 4
    assert 'roi' in deriv.entities
    assert 'subject' in deriv.entities
    preproc = layout_ds005_multi_derivs.get(desc='preproc')
    assert len(preproc) == 3


def test_layout_with_conflicting_deriv_folders():
    data_dir = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(data_dir)
    deriv_dir1 = join(get_test_data_path(), 'ds005_derivs', 'dummy')
    deriv_dir2 = join(get_test_data_path(), 'ds005_derivs', 'dummy')
    with pytest.raises(BIDSDerivativesValidationError):
        layout.add_derivatives([deriv_dir1, deriv_dir2])


def test_query_derivatives(layout_ds005_derivs):
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     extension='.tsv')
    result = [f.filename for f in result]
    assert len(result) == 49
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' in result
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     scope='raw', extension='.tsv')
    assert len(result) == 48
    result = [f.filename for f in result]
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' not in result
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     desc='extra', extension='.tsv')
    assert len(result) == 1
    result = [f.filename for f in result]
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' in result


def test_restricted_words_in_path(tmpdir):
    orig_path = join(get_test_data_path(), 'synthetic')
    parent_dir = str(tmpdir / 'derivatives' / 'pipeline')
    os.makedirs(parent_dir)
    new_path = join(parent_dir, 'sourcedata')
    os.symlink(orig_path, new_path)
    orig_layout = BIDSLayout(orig_path)
    new_layout = BIDSLayout(new_path)

    orig_files = set(f.replace(orig_path, '') for f in orig_layout.files)
    new_files = set(f.replace(new_path, '') for f in new_layout.files)
    assert orig_files == new_files


def test_derivative_getters():
    synth_path = join(get_test_data_path(), 'synthetic')
    bare_layout = BIDSLayout(synth_path, derivatives=False)
    full_layout = BIDSLayout(synth_path, derivatives=True)
    assert bare_layout.get_spaces() == []
    assert set(full_layout.get_spaces()) == {'MNI152NLin2009cAsym', 'T1w'}


def test_get_tr(layout_7t_trt):
    # Bad subject, should fail
    with pytest.raises(NoMatchError) as exc:
        layout_7t_trt.get_tr(subject="zzz")
    assert str(exc.value).startswith("No functional images")
    # There are multiple tasks with different TRs, so this should fail
    with pytest.raises(NoMatchError) as exc:
        layout_7t_trt.get_tr(subject=['01', '02'])
    assert str(exc.value).startswith("Unique TR")
    # This should work
    tr = layout_7t_trt.get_tr(subject=['01', '02'], acquisition="fullbrain")
    assert tr == 3.0
    tr = layout_7t_trt.get_tr(subject=['01', '02'], acquisition="prefrontal")
    assert tr == 4.0

    tr = layout_7t_trt.get_RepetitionTime()
    assert sum([t in tr for t in [3.0, 4.0]]) == 2


def test_to_df(layout_ds117):
    # Only filename entities
    df = layout_ds117.to_df()
    assert df.shape == (115, 12)
    target = {'datatype', 'fmap', 'run', 'path', 'acquisition', 'scans',
              'session', 'subject', 'suffix', 'task', 'proc', 'extension'}
    assert set(df.columns) == target
    assert set(df['subject'].dropna().unique()) == {'01', '02', 'emptyroom'}

    # Include metadata entities
    df = layout_ds117.to_df(metadata=True)
    assert df.shape == (115, 56)
    assert not ({'InstitutionAddress', 'TriggerChannelCount', 'EchoTime'} -
                set(df.columns))


# XXX 0.14: Add dot to extension (difficult to parametrize with module-scoped fixture)
def test_parse_file_entities_from_layout(layout_synthetic):
    layout = layout_synthetic
    filename = '/sub-03_ses-07_run-4_desc-bleargh_sekret.nii.gz'

    # Test with entities taken from bids config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'extension': '.nii.gz'}
    assert target == layout.parse_file_entities(filename, config='bids')
    config = Config.load('bids')
    assert target == layout.parse_file_entities(filename, config=[config])
    assert target == layout.parse_file_entities(filename, scope='raw')

    # Test with default scope--i.e., everything
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'desc': 'bleargh', 'extension': '.nii.gz'}
    assert target == layout.parse_file_entities(filename)
    # Test with only the fmriprep pipeline (which includes both configs)
    assert target == layout.parse_file_entities(filename, scope='fmriprep')
    assert target == layout.parse_file_entities(filename, scope='derivatives')

    # Test with only the derivative config
    target = {'desc': 'bleargh'}
    assert target == layout.parse_file_entities(filename, config='derivatives')


def test_deriv_indexing():
    data_dir = join(get_test_data_path(), 'ds005')
    deriv_dir = join(data_dir, 'derivatives', 'bbr')

    # missing dataset_description.json
    with pytest.warns(UserWarning):
        layout = BIDSLayout(data_dir, derivatives=deriv_dir)

    # Should work fine
    deriv_dir = join(data_dir, 'derivatives', 'events')
    layout = BIDSLayout(data_dir, derivatives=deriv_dir)
    assert layout.get(scope='derivatives')
    assert layout.get(scope='events')
    assert not layout.get(scope='nonexistent')


def test_path_arguments():
    data_dir = join(get_test_data_path(), 'ds005')
    deriv_dir = join(data_dir, 'derivatives', 'events')

    layout = BIDSLayout(Path(data_dir), derivatives=Path(deriv_dir))
    assert layout.get(scope='derivatives')
    assert layout.get(scope='events')
    assert not layout.get(scope='nonexistent')

    layout = BIDSLayout(Path(data_dir), derivatives=[Path(deriv_dir)])
    assert layout.get(scope='derivatives')
    assert layout.get(scope='events')
    assert not layout.get(scope='nonexistent')


def test_layout_in_scope(layout_ds005, layout_ds005_derivs):
    assert layout_ds005._in_scope(['all'])
    assert layout_ds005._in_scope('raw')
    assert layout_ds005._in_scope(['all', 'ignored'])
    assert not layout_ds005._in_scope(['derivatives', 'ignored'])

    deriv = layout_ds005_derivs.derivatives['events']
    assert deriv._in_scope('all')
    assert deriv._in_scope(['derivatives'])
    assert deriv._in_scope('events')
    assert not deriv._in_scope('raw')


def test_get_layouts_in_scope(layout_ds005_multi_derivs):
    l = layout_ds005_multi_derivs
    assert len(l._get_layouts_in_scope('all')) == 3
    assert len(l._get_layouts_in_scope('nonexistent')) == 0
    assert len(l._get_layouts_in_scope(['events', 'dummy'])) == 2
    assert len(l._get_layouts_in_scope(['derivatives'])) == 2
    assert len(l._get_layouts_in_scope('raw')) == 1
    self_scope = l._get_layouts_in_scope('self')
    assert len(self_scope) == 1
    assert self_scope == [l]


def test_get_dataset_description(layout_ds005_multi_derivs):
    l = layout_ds005_multi_derivs
    dd = l.get_dataset_description()
    assert isinstance(dd, dict)
    assert dd['Name'] == 'Mixed-gambles task'
    dd = l.get_dataset_description('all', True)
    assert isinstance(dd, list)
    assert len(dd) == 3
    names = {'Mixed-gambles task', 'Mixed-gambles task -- dummy derivative'}
    assert set([d['Name'] for d in dd]) == names


def test_indexed_file_associations(layout_7t_trt):
    img = layout_7t_trt.get(subject='01', run=1, suffix='bold', session='1',
                            acquisition='fullbrain', extension='.nii.gz')[0]
    assocs = img.get_associations()
    assert len(assocs) == 3
    targets = {
        os.path.join(layout_7t_trt.root,
                     'sub-01/ses-1/fmap/sub-01_ses-1_run-1_phasediff.nii.gz'),
        os.path.join(
            img.dirname,
            'sub-01_ses-1_task-rest_acq-fullbrain_run-1_physio.tsv.gz'),
        os.path.join(
            img.dirname,
            'sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.json'
        )
    }
    assert set([a.path for a in assocs]) == set(targets)

    # Test with parents included
    targets.add(os.path.join(layout_7t_trt.root, 'task-rest_acq-fullbrain_bold.json'))
    assocs = img.get_associations(include_parents=True)
    assert len(assocs) == 4
    assert set([a.path for a in assocs]) == set(targets)

    # Get the root-level JSON and check that its associations are correct
    js = [a for a in assocs if a.path.endswith('json')][1]
    assert len(js.get_associations()) == 40
    assert len(js.get_associations('Parent')) == 1
    assert len(js.get_associations('Metadata')) == 39
    assert not js.get_associations('InformedBy')


def test_layout_save(tmp_path, layout_7t_trt):
    layout_7t_trt.save(str(tmp_path / "f.sqlite"),
                       replace_connection=False)
    data_dir = join(get_test_data_path(), '7t_trt')
    layout = BIDSLayout(data_dir, database_path=str(tmp_path / "f.sqlite"))
    oldfies = set(layout_7t_trt.get(suffix='events', return_type='file'))
    newfies = set(layout.get(suffix='events', return_type='file'))
    assert oldfies == newfies


def test_indexing_tag_conflict():
    data_dir = join(get_test_data_path(), 'ds005_conflict')
    with pytest.raises(BIDSValidationError) as exc:
        layout = BIDSLayout(data_dir)
    assert str(exc.value).startswith("Conflicting values found")
    assert 'run' in str(exc.value)


def test_get_with_wrong_dtypes(layout_7t_trt):
    ''' Test automatic dtype sanitization. '''
    l = layout_7t_trt
    assert (l.get(run=1) == l.get(run='1') == l.get(run=np.int64(1)) ==
            l.get(run=[1, '15']) == l.get(run='01'))
    assert not l.get(run='not_numeric')
    assert l.get(session=1) == l.get(session='1')


def test_get_with_regex_search(layout_7t_trt):
    """ Tests that regex-based searching works. """
    l = layout_7t_trt

    # subject matches both '10' and '01'
    results = l.get(subject='1', session='1', task='rest', suffix='bold',
                    acquisition='fron.al', extension='.nii.gz',
                    regex_search=True)
    assert len(results) == 2

    # subject matches '10'
    results = l.get(subject='^1', session='1', task='rest', suffix='bold',
                    acquisition='fron.al', extension='.nii.gz',
                    regex_search=True, return_type='filename')
    assert len(results) == 1
    assert results[0].endswith('sub-10_ses-1_task-rest_acq-prefrontal_bold.nii.gz')


def test_get_with_regex_search_bad_dtype(layout_7t_trt):
    """ Tests that passing in a non-string dtype for an entity doesn't crash
    regexp-based searching (i.e., that implicit conversion is done
    appropriately). """
    l = layout_7t_trt
    results = l.get(subject='1', run=1, task='rest', suffix='bold',
                    acquisition='fullbrain', extension='.nii.gz',
                    regex_search=True)
    # Two runs (1 per session) for each of subjects '10' and '01'
    assert len(results) == 4


def test_get_with_invalid_filters(layout_ds005):
    l = layout_ds005
    # Raise error with suggestions
    with pytest.raises(ValueError, match='session'):
        l.get(subject='12', ses=True, invalid_filters='error')
    with pytest.raises(ValueError, match='session'):
        l.get(subject='12', ses=True)
    # Silently drop amazing
    res_without = l.get(subject='12', suffix='bold')
    res_drop = l.get(subject='12', suffix='bold', amazing='!!!',
                     invalid_filters='drop')
    assert res_without == res_drop
    assert len(res_drop) == 3
    # Retain amazing, producing empty set
    allow_res = l.get(subject='12', amazing=True, invalid_filters='allow')
    assert allow_res == []

    # assert warning when filters are passed in
    filters = {'subject': '1'}
    with pytest.raises(RuntimeError, match='You passed in filters as a dictionary'):
        l.get(filters=filters)
    # Correct call:
    l.get(**filters)


def test_get_with_query_constants_in_match_list(layout_ds005):
    l = layout_ds005
    get1 = l.get(subject='12', run=1, suffix='bold')
    get_none = l.get(subject='12', run=None, suffix='bold')
    get_any = l.get(subject='12', run=Query.ANY, suffix='bold')
    get1_and_none = l.get(subject='12', run=[None, 1], suffix='bold')
    get1_and_any = l.get(subject='12', run=[Query.ANY, 1], suffix='bold')
    get_none_and_any = l.get(subject='12', run=[Query.ANY, Query.NONE], suffix='bold')
    assert set(get1_and_none) == set(get1) | set(get_none)
    assert set(get1_and_any) == set(get1) | set(get_any)
    assert set(get_none_and_any) == set(get_none) | set(get_any)


def test_get_non_run_entity_with_query_constants_in_match_list(layout_ds005):
    l = layout_ds005
    get1 = l.get(subject='01', acquisition="MPRAGE", suffix='T1w')
    get_none = l.get(subject='01', acquisition=None, suffix='T1w')
    get_any = l.get(subject='01', acquisition=Query.ANY, suffix='T1w')
    get1_and_none = l.get(subject='01', acquisition=[None, "MPRAGE"], suffix='T1w')
    get1_and_any = l.get(subject='01', acquisition=[Query.ANY, "MPRAGE"], suffix='T1w')
    get_none_and_any = l.get(
        subject='01', acquisition=[Query.ANY, Query.NONE], suffix='T1w'
    )
    assert set(get1_and_none) == set(get1) | set(get_none)
    assert set(get1_and_any) == set(get1) | set(get_any)
    assert set(get_none_and_any) == set(get_none) | set(get_any)


def test_query_constants_work_on_extension(layout_ds005_no_validate):
    l = layout_ds005_no_validate
    get_both = l.get(subject='11', datatype='dwi', extension=Query.OPTIONAL)
    get_ext = l.get(subject='11', datatype='dwi', extension=Query.REQUIRED)
    get_no_ext = l.get(subject='11', datatype='dwi', extension=Query.NONE)
    assert len(get_both) == 2
    assert len(get_ext) == 1
    assert len(get_no_ext) == 1
    assert 'extension' in get_ext[0].get_entities()
    assert 'extension' not in get_no_ext[0].get_entities()


def test_load_layout(layout_synthetic_nodb, db_dir):
    db_path = str(db_dir / 'tmp_db')
    layout_synthetic_nodb.save(db_path)
    reloaded = BIDSLayout.load(db_path)
    assert sorted(layout_synthetic_nodb.get(return_type='file')) == \
        sorted(reloaded.get(return_type='file'))
    cm1 = layout_synthetic_nodb.connection_manager
    cm2 = reloaded.connection_manager
    for attr in ['root', 'absolute_paths', 'config', 'derivatives']:
        assert getattr(cm1.layout_info, attr) == getattr(cm2.layout_info, attr)


def test_load_layout_config_not_overwritten(layout_synthetic_nodb, tmpdir):
    modified_dataset_path = tmpdir/"modified"
    shutil.copytree(layout_synthetic_nodb.root, modified_dataset_path)

    # Save index
    db_path = str(tmpdir / 'tmp_db')
    BIDSLayout(modified_dataset_path).save(db_path)

    # Update dataset_description.json
    dataset_description = modified_dataset_path/"dataset_description.json"
    with dataset_description.open('r') as f:
        description = json.load(f)
    description["DatasetType"] = "derivative"
    description["GeneratedBy"] = [
        { "Name": "foo" }
    ]
    with dataset_description.open('w') as f:
        json.dump(description, f)

    # Reload
    db_layout = BIDSLayout(modified_dataset_path, database_path=db_path)
    fresh_layout = BIDSLayout(modified_dataset_path, validate=False)
    cm1 = db_layout.connection_manager
    cm2 = fresh_layout.connection_manager
    for attr in ['root', 'absolute_paths', 'derivatives']:
        assert getattr(cm1.layout_info, attr) == getattr(cm2.layout_info, attr)

    assert cm1.layout_info.config != cm2.layout_info.config


def test_padded_run_roundtrip(layout_ds005):
    for run in (1, "1", "01"):
        res = layout_ds005.get(subject="01", task="mixedgamblestask",
                               run=run, extension=".nii.gz")
        assert len(res) == 1
    boldfile = res[0]
    ents = boldfile.get_entities()
    assert isinstance(ents["run"], PaddedInt)
    assert ents["run"] == 1
    newpath = layout_ds005.build_path(ents, absolute_paths=False)
    assert newpath == "sub-01/func/sub-01_task-mixedgamblestask_run-01_bold.nii.gz"

@pytest.mark.parametrize(
    "fname", [
        "sub-01/anat/sub-01_T1w.nii.gz",
        ".datalad",
        "code",
        "sub-01/.datalad",
    ],
)
def test_indexer_patterns(fname):
    root = Path("/home/user/.cache/data/")
    path = root / fname

    assert bool(_check_path_matches_patterns(
        path,
        [_regexfy("code")],
        root=root,
    )) is (fname == "code")

    assert bool(_check_path_matches_patterns(
        path,
        [_regexfy("code", root=root)],
        root=root,
    )) is (fname == "code")

    assert _check_path_matches_patterns(
        path,
        [re.compile(r"/\.")],
        root=None,
    ) is True

    assert _check_path_matches_patterns(
        path,
        [re.compile(r"/\.")],
        root=root,
    ) is (".datalad" in fname)
