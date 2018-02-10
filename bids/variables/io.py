import numpy as np
import pandas as pd
import nibabel as nb
from os.path import join
from bids.utils import listify
from .entities import NodeIndex
from .variables import SparseRunVariable, DenseRunVariable, SimpleVariable
import warnings


BASE_ENTITIES = ['subject', 'session', 'task', 'run']
ALL_ENTITIES = BASE_ENTITIES + ['modality', 'type', 'acq']


def load_variables(layout, types=None, levels=None, skip_empty=True, **kwargs):
    ''' A convenience wrapper for one or more load_*_variables() calls.

    Args:
        layout (BIDSLayout): BIDSLayout containing variable files.
        types (str, list): Types of variables to retrieve. All valid values
            reflect the filename stipulated in the BIDS spec for each kind of
            variable. Valid values include: 'events', 'physio', 'stim',
            'scans', 'participants', 'sessions', and 'confounds'.
        levels (str, list): Optional level(s) of variables to load. Valid
            values are 'run', 'session', 'subject', or 'dataset'. This is
            simply a shorthand way to specify types--e.g., 'run' will be
            converted to types=['events', 'physio', 'stim', 'confounds'].
        skip_empty (bool): Whether or not to skip empty Variables (i.e.,
            where there are no rows/records in a file after applying any
            filtering operations like dropping NaNs).
        kwargs: Optional keyword arguments to pass onto the individual
            load_*_variables() calls.

    Returns:
        A NodeIndex instance.

    Example:
        >>> load_variables(layout, ['events', 'physio'], subject='01')
        # returns all variables stored in _events.tsv and _physio.tsv.gz files
        # for runs that belong to subject with id '01'.
    '''

    TYPES = ['events', 'physio', 'stim', 'scans', 'participants', 'sessions',
             'confounds']

    types = listify(types)

    if types is None:
        if levels is not None:
            types = []
            lev_map = {
                'run': ['events', 'physio', 'stim', 'confounds'],
                'session': ['scans'],
                'subject': ['sessions'],
                'dataset': ['participants']
            }
            [types.extend(lev_map[l]) for l in listify(levels)]
        else:
            types = TYPES

    bad_types = set(types) - set(TYPES)
    if bad_types:
        raise ValueError("Invalid variable types: %s" % bad_types)

    dataset = NodeIndex()

    run_types = list({'events', 'physio', 'stim', 'confounds'} - set(types))
    type_flags = {t: False for t in run_types}
    if len(type_flags) < 4:
        _kwargs = kwargs.copy()
        _kwargs.update(type_flags)
        dataset = _load_time_variables(layout, dataset, **_kwargs)

    for t in ({'scans', 'sessions', 'participants'} & set(types)):
        dataset = _load_tsv_variables(layout, t, dataset, **kwargs)

    return dataset


def _load_time_variables(layout, dataset=None, columns=None, scan_length=None,
                         drop_na=True, events=True, physio=True, stim=True,
                         confounds=True, derivatives=None, skip_empty=True,
                         **selectors):
    ''' Loads all variables found in *_events.tsv files and returns them as a
    BIDSVariableCollection.

    Args:
        layout (BIDSLayout): A BIDSLayout to scan.
        dataset (NodeIndex): A BIDS NodeIndex container. If None, a new one is
            initialized.
        columns (list): Optional list of names specifying which columns in the
            event files to read. By default, reads all columns found.
        scan_length (float): Optional duration of runs (in seconds). By
            default, this will be extracted from the BOLD image. However, in
            cases where the user doesn't have access to the images (e.g.,
            because only file handles are locally available), a fixed duration
            can be manually specified as a fallback.
        drop_na (bool): If True, removes all events where amplitude is n/a. If
            False, leaves n/a values intact. Note that in the latter case,
            transformations that requires numeric values may fail.
        events (bool): If True, extracts variables from events.tsv
            files.
        physio (bool): If True, extracts variables from _physio files.
        stim (bool): If True, extracts variables from _stim files.
        derivatives (str): How to handle derivative events. Passed to
            BIDSLayout.get_events.
        skip_empty (bool): Whether or not to skip empty Variables (i.e.,
            where there are no rows/records in a file, or all onsets,
            durations, and amplitudes are 0).
        selectors (dict): Optional keyword arguments passed onto the
            BIDSLayout instance's get() method; can be used to constrain
            which data are loaded.

    Returns: A NodeIndex instance.
    '''

    selectors = {k: v for k, v in selectors.items() if k in BASE_ENTITIES}

    if dataset is None:
        dataset = NodeIndex()

    images = layout.get(return_type='file', type='bold', modality='func',
                        extensions='.nii.gz', **selectors)

    if not images:
        raise ValueError("No functional images that match criteria found.")

    # Main loop over images
    for img_f in images:

        entities = layout.files[img_f].entities

        # Run is not mandatory, but we need a default for proper indexing
        if 'run' in entities:
            entities['run'] = int(entities['run'])

        # Get duration of run: first try to get it directly from the image
        # header; if that fails, try to get NumberOfVolumes from the
        # run metadata; if that fails, look for a scan_length argument.
        try:
            img = nb.load(img_f)
            duration = img.shape[3] * img.header.get_zooms()[-1]
        except Exception as e:
            if scan_length is not None:
                duration = scan_length
            else:
                msg = ("Unable to extract scan duration from one or more "
                       "BOLD runs, and no scan_length argument was provided "
                       "as a fallback. Please check that the image files are "
                       "available, or manually specify the scan duration.")
                raise ValueError(msg)

        tr = layout.get_metadata(img_f)['RepetitionTime']

        run = dataset.get_or_create_node('run', entities, image_file=img_f,
                                         duration=duration, repetition_time=tr)
        run_info = run.get_info()

        # Process event files
        if events:
            _data = layout.get_events(img_f, return_type='df',
                                      derivatives=derivatives)
            if _data is not None:

                if 'amplitude' in _data.columns:
                    if (_data['amplitude'].astype(int) == 1).all() and \
                            'trial_type' in _data.columns:
                        msg = ("Column 'amplitude' with constant value 1 is "
                               "unnecessary in event files; ignoring it.")
                        _data = _data.drop('amplitude', axis=1)
                    else:
                        msg = ("Column name 'amplitude' is reserved; renaming "
                               "it to 'amplitude_'.")
                        _data = _data.rename(
                            columns={'amplitude': 'amplitude_'})
                    warnings.warn(msg)

                _data = _data.replace('n/a', np.nan)  # Replace BIDS' n/a
                _data = _data.apply(pd.to_numeric, errors='ignore')

                _cols = columns or list(set(_data.columns.tolist()) -
                                        {'onset', 'duration'})

                # Construct a DataFrame for each extra column
                for col in _cols:
                    df = _data[['onset', 'duration']].copy()
                    df['amplitude'] = _data[col].values

                    # Add in all of the run's entities as new columns for index
                    for entity, value in entities.items():
                        if entity in BASE_ENTITIES:
                            df[entity] = value

                    if drop_na:
                        df = df.dropna(subset=['amplitude'])

                    if df.empty:
                        continue

                    var = SparseRunVariable(col, df, run_info, 'events')
                    run.add_variable(var)

        # Process confound files
        if confounds:
            sub_ents = {k: v for k, v in entities.items()
                        if k in BASE_ENTITIES}
            confound_files = layout.get(type='confounds', **sub_ents)
            for cf in confound_files:
                _data = pd.read_csv(cf.filename, sep='\t', na_values='n/a')
                if columns is not None:
                    conf_cols = list(set(_data.columns) & set(columns))
                    _data = _data.loc[:, conf_cols]
                for col in _data.columns:
                    sr = 1. / run.repetition_time
                    var = DenseRunVariable(col, _data[[col]], run_info,
                                           'confounds', sr)
                    run.add_variable(var)

        # Process recordinging files
        if physio or stim:
            rec_types = ['physio'] if physio else []
            if stim:
                rec_types.append('stim')
            rec_files = layout.get_nearest(img_f, extensions='.tsv.gz',
                                           all_=True, type=rec_types,
                                           ignore_strict_entities=['type'])
            for rf in rec_files:
                metadata = layout.get_metadata(rf)
                if not metadata:
                    raise ValueError("No .json sidecar found for '%s'." % rf)
                data = pd.read_csv(rf, sep='\t')
                freq = metadata['SamplingFrequency']
                st = metadata['StartTime']
                rf_cols = metadata['Columns']
                data.columns = rf_cols

                # Filter columns if user passed names
                if columns is not None:
                    rf_cols = list(set(rf_cols) & set(columns))
                    data = data.loc[:, rf_cols]

                n_cols = len(rf_cols)
                if not n_cols:
                    continue

                # Keep only in-scan samples
                if st < 0:
                    start_ind = np.floor(-st * freq)
                    values = data.values[start_ind:, :]
                else:
                    values = data.values

                if st > 0:
                    n_pad = freq * st
                    pad = np.zeros((n_pad, n_cols))
                    values = np.r_[pad, values]

                n_rows = int(run.duration * freq)
                if len(values) > n_rows:
                    values = values[:n_rows, :]
                elif len(values) < n_rows:
                    pad = np.zeros((n_rows - len(values), n_cols))
                    values = np.r_[values, pad]

                df = pd.DataFrame(values, columns=rf_cols)
                source = 'physio' if '_physio.tsv' in rf else 'stim'
                for col in df.columns:
                    var = DenseRunVariable(col, df[[col]], run_info, source,
                                           freq)
                    run.add_variable(var)
    return dataset


def _load_tsv_variables(layout, type_, dataset=None, columns=None,
                        prepend_type=False, **selectors):
    ''' Reads variables from scans.tsv, sessions.tsv, and participants.tsv.

    Args:
        layout (BIDSLayout): The BIDSLayout to use.
        type_ (str): The type of file to read from. Must be one of 'scans',
            'sessions', or 'participants'.
        dataset (NodeIndex): A BIDS NodeIndex container. If None, a new one is
            initialized.
        columns (list): Optional list of names specifying which columns in the
            files to return. If None, all columns are returned.
        prepend_type (bool): If True, variable names are prepended with the
            type name (e.g., 'age' becomes 'participants.age').
        selectors (dict): Optional keyword arguments passed onto the
            BIDSLayout instance's get() method; can be used to constrain
            which data are loaded.

    Returns: A NodeIndex instance.
    '''

    # Sanitize the selectors: only keep entities at current level or above
    remap = {'scans': 'run', 'sessions': 'session', 'participants': 'subject'}
    level = remap[type_]
    valid_entities = BASE_ENTITIES[:BASE_ENTITIES.index(level)]
    layout_kwargs = {k: v for k, v in selectors.items() if k in valid_entities}

    if dataset is None:
        dataset = NodeIndex()

    files = layout.get(extensions='.tsv', return_type='file', type=type_,
                       **layout_kwargs)

    for f in files:

        f = layout.files[f]
        _data = pd.read_table(f.path, sep='\t')

        # Entities can be defined either within the first column of the .tsv
        # file (for entities that vary by row), or from the full file path
        # (for entities constant over all rows in the file). We extract both
        # and store them in the main DataFrame alongside other variables (as
        # they'll be extracted when the Column is initialized anyway).
        for ent_name, ent_val in f.entities.items():
            if ent_name in BASE_ENTITIES:
                _data[ent_name] = ent_val

        # Handling is a bit more convoluted for scans.tsv, because the first
        # column contains the run filename, which we also need to parse.
        if type_ == 'scans':
            image = _data['filename']
            _data = _data.drop('filename', axis=1)
            dn = f.dirname
            paths = [join(dn, p) for p in image.values]
            ent_recs = [layout.files[p].entities for p in paths
                        if p in layout.files]
            ent_cols = pd.DataFrame.from_records(ent_recs)
            _data = pd.concat([_data, ent_cols], axis=1)
            # It's possible to end up with duplicate entity columns this way
            _data = _data.T.drop_duplicates().T

        # The BIDS spec requires ID columns to be named 'session_id', 'run_id',
        # etc., and IDs begin with entity prefixes (e.g., 'sub-01'). To ensure
        # consistent internal handling, we strip these suffixes and prefixes.
        elif type_ == 'sessions':
            _data = _data.rename(columns={'session_id': 'session'})
            _data['session'] = _data['session'].str.replace('ses-', '')
        elif type_ == 'participants':
            _data = _data.rename(columns={'participant_id': 'subject'})
            _data['subject'] = _data['subject'].str.replace('sub-', '')

        # Filter rows on all selectors
        comm_cols = list(set(_data.columns) & set(selectors.keys()))
        for col in comm_cols:
            vals = listify(selectors.get(col))
            _data = _data.query('%s in @vals' % col)

        level = {'scans': 'session', 'sessions': 'subject',
                 'participants': 'dataset'}[type_]
        node = dataset.get_or_create_node(level, f.entities)

        ent_cols = list(set(ALL_ENTITIES) & set(_data.columns))
        amp_cols = list(set(_data.columns) - set(ent_cols))

        if columns is not None:
            amp_cols = list(set(amp_cols) & set(columns))

        for col_name in amp_cols:

            # Rename colummns: values must be in 'amplitude'
            df = _data.loc[:, [col_name] + ent_cols]
            df.columns = ['amplitude'] + ent_cols

            if prepend_type:
                col_name = '%s.%s' % (type_, col_name)

            node.add_variable(SimpleVariable(col_name, df, type_))

    return dataset
