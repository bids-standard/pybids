""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """

import os
import pytest
from bids.layout import BIDSLayout, parse_file_entities
from bids.layout.core import BIDSFile, Entity, Config
from os.path import join, abspath, basename
from bids.tests import get_test_data_path


# Fixture uses in the rest of the tests
@pytest.fixture(scope='module')
def layout_7t_trt():
    data_dir = join(get_test_data_path(), '7t_trt')
    return BIDSLayout(data_dir)


@pytest.fixture(scope='module')
def layout_ds005():
    data_dir = join(get_test_data_path(), 'ds005')
    return BIDSLayout(data_dir)


@pytest.fixture(scope='module')
def layout_ds117():
    data_dir = join(get_test_data_path(), 'ds000117')
    return BIDSLayout(data_dir)


@pytest.fixture(scope='module')
def layout_ds005_derivs():
    data_dir = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(data_dir)
    deriv_dir = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives(deriv_dir)
    return layout


@pytest.fixture(scope='module')
def layout_ds005_multi_derivs():
    data_dir = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(data_dir)
    deriv_dir1 = join(get_test_data_path(), 'ds005_derivs')
    deriv_dir2 = join(data_dir, 'derivatives', 'events')
    layout.add_derivatives([deriv_dir1, deriv_dir2])
    return layout


@pytest.fixture(scope='module')
def layout_ds005_models():
    data_dir = join(get_test_data_path(), 'ds005')
    return BIDSLayout(data_dir, validate=True, force_index=['models'])

@pytest.fixture(scope='module')
def layout_synthetic():
    path = join(get_test_data_path(), 'synthetic')
    return BIDSLayout(path, derivatives=True)


def test_layout_init(layout_7t_trt):
    assert isinstance(layout_7t_trt.files, dict)


def test_layout_repr(layout_7t_trt):
    assert "Subjects: 10 | Sessions: 20 | Runs: 20" in str(layout_7t_trt)


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
    target = (layout.root + '/' +  orig_file).split('/')
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
    target = (layout.root + '/' +  orig_file).split('/')
    target = os.path.sep + os.path.join(*target)
    assert layout.get_file(target)
    assert not layout.get_file(target, scope='raw')
    assert layout.get_file(target, scope='derivatives')
    assert layout.get_file(target, scope='events')

    # No such file
    assert not layout.get_file('bleargh')
    assert not layout.get_file('/absolute/bleargh')


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
    result = layout_7t_trt.get_metadata(join(layout_7t_trt.root, *target),
                                      include_entities=True)
    assert result['EchoTime'] == 0.020
    assert result['subject'] == '01'
    assert result['acquisition'] == 'fullbrain'


def test_get_metadata_via_bidsfile(layout_7t_trt):
    ''' Same as test_get_metadata5, but called through BIDSFile. '''
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    target = target.split('/')
    path = join(layout_7t_trt.root, *target)
    result = layout_7t_trt.files[path].metadata
    assert result['EchoTime'] == 0.020
    # include_entities is False when called through a BIDSFile
    assert 'subject' not in result


def test_get_with_bad_target(layout_7t_trt):
    with pytest.raises(ValueError) as exc:
        layout_7t_trt.get(target='unicorn')
        msg = exc.value.message
        assert 'subject' in msg and 'reconstruction' in msg and 'proc' in msg
    with pytest.raises(ValueError) as exc:
        layout_7t_trt.get(target='sub')
        msg = exc.value.message
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


def test_force_index(layout_ds005, layout_ds005_models):
    target= join(layout_ds005_models.root, 'models',
                'ds-005_type-test_model.json')
    assert target not in layout_ds005.files
    assert target in layout_ds005_models.files
    assert 'all' not in layout_ds005_models.get_subjects()
    for f in layout_ds005_models.files.values():
        assert 'derivatives' not in f.path


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
    result = layout_ds005_derivs.get(suffix='events', return_type='object')
    result = [f.filename for f in result]
    assert len(result) == 49
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' in result
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     scope='raw')
    assert len(result) == 48
    result = [f.filename for f in result]
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' not in result
    result = layout_ds005_derivs.get(suffix='events', return_type='object',
                                     desc='extra')
    assert len(result) == 1
    result = [f.filename for f in result]
    assert 'sub-01_task-mixedgamblestask_run-01_desc-extra_events.tsv' in result


def test_get_bidsfile_image_prop():
    path = "synthetic/sub-01/ses-01/func/sub-01_ses-01_task-nback_run-01_bold.nii.gz"
    path = path.split('/')
    path = join(get_test_data_path(), *path)
    bf = BIDSFile(path, None)
    assert bf.image is not None
    assert bf.image.shape == (64, 64, 64, 64)


def test_restricted_words_in_path(tmpdir):
    orig_path = join(get_test_data_path(), 'synthetic')
    parent_dir = str(tmpdir / 'derivatives' / 'pipeline')
    os.makedirs(parent_dir)
    new_path = join(parent_dir, 'sourcedata')
    os.symlink(orig_path, new_path)
    print(orig_path, new_path)
    orig_layout = BIDSLayout(orig_path)
    new_layout = BIDSLayout(new_path)

    orig_files = set(f.replace(orig_path, '') for f in orig_layout.files)
    new_files = set(f.replace(new_path, '') for f in new_layout.files)
    assert orig_files == new_files


def test_derivative_getters():
    synth_path = join(get_test_data_path(), 'synthetic')
    bare_layout = BIDSLayout(synth_path, derivatives=False)
    full_layout = BIDSLayout(synth_path, derivatives=True)
    with pytest.raises(AttributeError):
        bare_layout.get_spaces()
    assert set(full_layout.get_spaces()) == {'MNI152NLin2009cAsym', 'T1w'}


def test_get_tr(layout_7t_trt):
    # Bad subject, should fail
    with pytest.raises(ValueError) as exc:
        layout_7t_trt.get_tr(subject="zzz")
        assert exc.value.message.startswith("No functional images")
    # There are multiple tasks with different TRs, so this should fail
    with pytest.raises(ValueError) as exc:
        layout_7t_trt.get_tr(subject=['01', '02'])
        assert exc.value.message.startswith("Unique TR")
    # This should work
    tr = layout_7t_trt.get_tr(subject=['01', '02'], acquisition="fullbrain")
    assert tr == 3.0
    tr = layout_7t_trt.get_tr(subject=['01', '02'], acquisition="prefrontal")
    assert tr == 4.0


def test_parse_file_entities():
    filename = '/sub-03_ses-07_run-4_desc-bleargh_sekret.nii.gz'

    # Test with entities taken from bids config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret'}
    assert target == parse_file_entities(filename, config='bids')
    config = Config.load('bids')
    assert target == parse_file_entities(filename, config=[config])

    # Test with entities taken from bids and derivatives config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'desc': 'bleargh'}
    assert target == parse_file_entities(filename)
    assert target == parse_file_entities(filename, config=['bids', 'derivatives'])

    # Test with list of Entities
    entities = [
        Entity('subject', "[/\\\\]sub-([a-zA-Z0-9]+)"),
        Entity('run', "[_/\\\\]run-0*(\\d+)", dtype=int),
        Entity('suffix', "[._]*([a-zA-Z0-9]*?)\\.[^/\\\\]+$"),
        Entity('desc', "desc-([a-zA-Z0-9]+)"),
    ]
    # Leave out session to distinguish from previous test target
    target = {'subject': '03', 'run': 4, 'suffix': 'sekret', 'desc': 'bleargh'}
    assert target == parse_file_entities(filename, entities=entities)


def test_parse_file_entities_from_layout(layout_synthetic):
    layout = layout_synthetic
    filename = '/sub-03_ses-07_run-4_desc-bleargh_sekret.nii.gz'

    # Test with entities taken from bids config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret'}
    assert target == layout.parse_file_entities(filename, config='bids')
    config = Config.load('bids')
    assert target == layout.parse_file_entities(filename, config=[config])
    assert target == layout.parse_file_entities(filename, scope='raw')

    # Test with default scope—i.e., everything
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'desc': 'bleargh'}
    assert target == layout.parse_file_entities(filename)
    # Test with only the fmriprep pipeline (which includes both configs)
    assert target == layout.parse_file_entities(filename, scope='fmriprep')
    assert target == layout.parse_file_entities(filename, scope='derivatives')

    # Test with only the derivative config
    target = {'desc': 'bleargh'}
    assert target == layout.parse_file_entities(filename, config='derivatives')