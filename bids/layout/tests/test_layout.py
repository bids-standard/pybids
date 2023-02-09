""" Tests of functionality in the layout module--mostly related to the
BIDSLayout class."""

import json
import os
import re
import shutil
from os.path import join, abspath, basename
from pathlib import Path

import numpy as np
import pytest

from bids.exceptions import (
    BIDSDerivativesValidationError,
    BIDSValidationError,
    NoMatchError,
    TargetError,
)
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
from bids.utils import natural_sort


def test_layout_init(layout_7t_trt):
    assert isinstance(layout_7t_trt.files, dict)


@pytest.mark.parametrize(
    'index_metadata,query,result',
    [
        (True, {}, 3.0),
        (True, {}, 3.0),
        (True, {'task': 'rest'}, 3.0),
        (True, {'task': 'rest', 'extension': ['.nii.gz']}, 3.0),
        (True, {'task': 'rest', 'extension': '.nii.gz'}, 3.0),
        (True, {'task': 'rest', 'extension': ['.nii.gz', '.json'], 'return_type': 'file'}, 3.0),
    ])
def test_file_get_metadata(index_metadata, query, result, mock_config):
    data_dir = join(get_test_data_path(), '7t_trt')
    layout = BIDSLayout(data_dir, index_metadata=index_metadata, **query)
    sample_file = layout.get(task='rest', extension='.nii.gz',
                             acquisition='fullbrain')[0]
    metadata = sample_file.get_metadata()
    assert metadata.get('RepetitionTime') == result


def test_layout_repr(layout_7t_trt):
    assert "Subjects: 10 | Sessions: 2 | Runs: 2" in str(layout_7t_trt)


def test_invalid_dataset_description(tmp_path):
    shutil.copytree(join(get_test_data_path(), '7t_trt'), tmp_path / "7t_dset")
    (tmp_path / "7t_dset" / "dataset_description.json").write_text(
        "I am not a valid json file"
    )
    with pytest.raises(BIDSValidationError) as exc:
        BIDSLayout(tmp_path / "7t_dset")


def test_layout_repr_overshadow_run(tmp_path):
    """A test creating a layout to replicate #681."""
    shutil.copytree(join(get_test_data_path(), '7t_trt'), tmp_path / "7t_trt")
    (tmp_path / "7t_trt" / "sub-01" / "ses-1" / "sub-01_ses-1_scans.json").write_text(
        json.dumps({"run": {"Description": "metadata to cause #681"}})
    )
    assert "Subjects: 10 | Sessions: 2 | Runs: 2" in str(BIDSLayout(tmp_path / "7t_trt"))


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
    target = os.path.join(layout.root, *orig_file.split('/'))
    assert layout.get_file(target)
    assert layout.get_file(target, scope='raw')
    assert not layout.get_file(target, scope='derivatives')

    # relative path in derivatives pipeline
    orig_file = 'events/sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv'
    target = os.path.join(*orig_file.split('/'))
    assert not layout.get_file(target)
    assert layout.get_file(target, scope='derivatives')

    # absolute path in derivatives pipeline
    orig_file = 'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv'
    target = os.path.join(*orig_file.split('/'))
    assert not layout.get_file(target)
    assert not layout.get_file(target, scope='derivatives')
    assert layout.get_file(target, scope='derivatives/events')

    # No such file
    assert not layout.get_file('bleargh')
    assert not layout.get_file('/absolute/bleargh')


class TestDerivativeAsRoot:
    def test_dataset_without_datasettype_parsed_as_raw(self):
        dataset_path = Path("ds005_derivs", "format_errs", "no_dataset_type")
        unvalidated = BIDSLayout(
            Path(get_test_data_path()) / dataset_path,
            validate=False
        )
        assert len(unvalidated.get()) == 4
        with pytest.raises(ValueError):
            unvalidated.get(desc="preproc")

        validated = BIDSLayout(Path(get_test_data_path()) / dataset_path)
        assert len(validated.get()) == 1

    def test_dataset_missing_generatedby_fails_validation(self):
        dataset_path = Path("ds005_derivs", "format_errs", "no_pipeline_description")
        with pytest.raises(BIDSDerivativesValidationError):
            BIDSLayout(Path(get_test_data_path()) / dataset_path)

    def test_correctly_formatted_derivative_loads_as_derivative(self):
        dataset_path = Path("ds005_derivs", "dummy")
        layout = BIDSLayout(Path(get_test_data_path()) / dataset_path)
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
            Path(get_test_data_path()) / dataset_path,
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

    result = layout_7t_trt.get_metadata(path)
    with pytest.raises(KeyError) as err:
        result['Missing']


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


def test_get_return_type_dir(layout_7t_trt):
    res_relpath = layout_7t_trt.get(target='sub', return_type='dir')
    target_relpath = ["sub-{:02d}".format(i) for i in range(1, 11)]
    assert all([tp in res_relpath for tp in target_relpath])


def test_get_val_none(layout_7t_trt):
    t1w_files = layout_7t_trt.get(subject='01', session='1', suffix='T1w')
    assert len(t1w_files) == 1
    assert 'acq' not in t1w_files[0].name
    t1w_files = layout_7t_trt.get(
        subject='01', session='1', suffix='T1w', acquisition=None)
    assert len(t1w_files) == 1
    bold_files = layout_7t_trt.get(
        subject='01', session='1', suffix='bold', acquisition=None)
    assert len(bold_files) == 0


def test_get_val_enum_any(layout_7t_trt):
    t1w_files = layout_7t_trt.get(
        subject='01', session='1', suffix='T1w', acquisition="*",
        extension='*')
    assert not t1w_files
    bold_files = layout_7t_trt.get(subject='01', session='1', run=1,
                                   suffix='bold', acquisition="*")
    assert len(bold_files) == 2


def test_get_val_enum_any_optional(layout_7t_trt, layout_ds005):
    # layout with sessions
    bold_files = layout_7t_trt.get(suffix='bold', run=1, subject='01')
    assert len(bold_files) == 3

    # layout without sessions
    bold_files = layout_ds005.get(suffix='bold', run=1, subject='01', session='*')
    assert not bold_files
    bold_files = layout_ds005.get(suffix='bold', run=1, subject='01')
    assert len(bold_files) == 1


def test_get_return_sorted(layout_7t_trt):
    paths = layout_7t_trt.get(target='sub', return_type='file')
    assert natural_sort(paths) == paths


def test_layout_with_derivs(layout_ds005_derivs):
    assert layout_ds005_derivs.root == join(get_test_data_path(), 'ds005')
    assert isinstance(layout_ds005_derivs.files, dict)
    assert len(layout_ds005_derivs.derivatives) == 1
    deriv = layout_ds005_derivs.derivatives['events']
    files = deriv.query()
    event_file = "sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv"
    deriv_files = [f.name for f in files]
    assert event_file in deriv_files
    entities = deriv.query_entities()
    assert 'sub' in entities


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


def test_query_derivatives(layout_ds005_derivs):
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     extension='.tsv')
    result = [f.name for f in result]
    assert len(result) == 49
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' in result
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     scope='raw', extension='.tsv')
    assert len(result) == 48
    result = [f.name for f in result]
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' not in result
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     desc='extra', extension='.tsv')
    assert len(result) == 1
    result = [f.name for f in result]
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' in result


def test_derivative_getters():
    synth_path = join(get_test_data_path(), 'synthetic')
    full_layout = BIDSLayout(synth_path)
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


def test_path_arguments():
    data_dir = join(get_test_data_path(), 'ds005')
    deriv_dir = join(data_dir, 'derivatives', 'events')

    layout = BIDSLayout(Path(data_dir), validate=False)
    assert layout.get(scope='derivatives/events')
    assert not layout.get(scope='nonexistent')


def test_get_dataset_description(layout_ds005_derivs):
    dd = layout_ds005_derivs.get_dataset_description()
    assert isinstance(dd, dict)
    assert dd['Name'] == 'Mixed-gambles task'
    dd = layout_ds005_derivs.get_dataset_description('all', True)
    assert isinstance(dd, list)
    assert len(dd) == 2
    names = {'Mixed-gambles task'}
    assert set([d['Name'] for d in dd]) == names


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
    get_any = l.get(subject='12', run='*', suffix='bold')
    get1_and_none = l.get(subject='12', run=[None, 1], suffix='bold')
    get1_and_any = l.get(subject='12', run=['*', 1], suffix='bold')
    get_none_and_any = l.get(subject='12', run=['*', None], suffix='bold')
    assert set(get1_and_none) == set(get1) | set(get_none)
    assert set(get1_and_any) == set(get1) | set(get_any)
    assert set(get_none_and_any) == set(get_none) | set(get_any)


def test_padded_run_roundtrip(layout_ds005):
    for run in (1, "1", "01"):
        res = layout_ds005.get(subject="01", task="mixedgamblestask",
                               run=run, extension=".nii.gz")
        assert len(res) == 1
    boldfile = res[0]
    ents = boldfile.get_entities()
    assert isinstance(ents["run"], int)
    assert ents["run"] == 1
    # TODO buld_path() not supported yet
    # newpath = layout_ds005.build_path(ents, absolute_paths=False)
    # assert newpath == "sub-01/func/sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
