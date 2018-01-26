import numpy as np
import pandas as pd
import nibabel as nb
import warnings
from collections import namedtuple
import math
from copy import copy, deepcopy
from abc import abstractproperty, abstractmethod, ABCMeta
import re
from bids.utils import listify
from scipy.interpolate import interp1d
from pandas.api.types import is_numeric_dtype
from os.path import dirname, join


BASE_ENTITIES = ['subject', 'session', 'task', 'run']


def load_event_variables(layout, entities=None, columns=None, scan_length=None,
                         sampling_rate=10, drop_na=True, extract_events=True,
                         extract_recordings=True, interp_method='linear',
                         extract_confounds=True, derivatives=None,
                         **selectors):
    ''' Loads all variables found in *_events.tsv files and returns them as a
    BIDSVariableCollection.

    Args:
        layouts (str, list): Path(s) to the root of the BIDS project(s). Can
            also be a list of Layouts.
        entities (list): Optional list of entities to encode and index by.
            If None (default), uses the standard BIDS-defined entities of
            ['run', 'session', 'subject', 'task'].
        columns (list): Optional list of names specifying which columns in the
            event files to read. By default, reads all columns found.
        scan_length (float): Optional duration of runs (in seconds). By
            default, this will be extracted from the BOLD image. However, in
            cases where the user doesn't have access to the images (e.g.,
            because only file handles are locally available), a fixed duration
            can be manually specified as a fallback.
        sampling_rate (int, float): Sampling rate to use internally when
            converting sparse columns to dense format.
        drop_na (bool): If True, removes all events where amplitude is n/a. If
            False, leaves n/a values intact. Note that in the latter case,
            transformations that requires numeric values may fail.
        extract_events (bool): If True, extracts variables from events.tsv
            files.
        extract_recordings (bool): If True, extracts variables from physio and
            stim recording files with dense sampling.
        interp_method (str): Interpolation method to use when resampling
            recording files to the desired sampling rate. Can be any value
            recognized by scipy.interpolate.interp1d.
        derivatives (str): How to handle derivative events. Passed to
            BIDSLayout.get_events.
        selectors (dict): Optional keyword arguments passed onto the
            BIDSLayout instance's get() method; can be used to constrain
            which data are loaded.

    Returns: A BIDSEventVariableCollection.
    '''

    images = layout.get(return_type='file', type='bold', modality='func',
                        extensions='.nii.gz', **selectors)

    if not images:
        raise ValueError("No functional images that match criteria found.")

    # Make sure all images have a common TR. In principle we could support
    # varying TRs, but this would make life much more complicated, and in any
    # case it almost invariably means the user is trying to model more than one
    # task at the same time, which is usually a mistake.
    run_trs = [layout.get_metadata(img)['RepetitionTime'] for img in images]
    if len(set(run_trs)) > 1:
        raise ValueError("More than one TR detected across specified runs."
                         " Currently we can only handle sets of runs that "
                         "share the same repetition time.")
    tr = 1. / run_trs[0]

    # Valid entities
    if entities is None:
        entities = ['run', 'session', 'subject', 'task', 'recording']

    # # Helper function to consolidate entities. Because of inheritance,
    # # tsv filenames don't always contain all entities, so we first get
    # # them from the image file, then update with anything found in the tsv.
    # def _combine_entities(img_path, tsv_path):
    #     ents = layout.files[img_path].entities.copy()
    #     ents.update(layout.files[tsv_path].entities)
    #     return {k: v for k, v in ents.items() if k in entities}

    # Initialize counters/storage vars
    run_start_time = 0
    event_dfs = []
    confound_dfs = []
    rec_dfs = []

    # Initialize collection
    collection = BIDSEventVariableCollection('time', entities=entities,
                                             sampling_rate=sampling_rate,
                                             repetition_time=tr)

    def get_entities(img):
        f_ents = layout.files[img].entities.copy()
        f_ents['unique_run_id'] = len(collection.run_infos)
        return f_ents

    # Main loop over images
    for img_f in images:

        # Get duration of run: first try to get it directly from the image
        # header; if that fails, try to get NumberOfVolumes from the
        # run metadata; if that fails, look for a scan_length argument.
        try:
            img = nb.load(img_f)
            duration = img.shape[3] * img.header.get_zooms()[-1]
        except Exception as e:
            try:
                md = layout.get_metadata(img_f)
                duration = md['NumberOfVolumes'] * md['RepetitionTime']
            except Exception as e:
                if scan_length is not None:
                    duration = scan_length
                else:
                    msg = ("Unable to extract scan duration from one or more "
                       "BOLD runs, and no scan_length argument was provided "
                       "as a fallback. Please check that the image files are "
                       "available, or manually specify the scan duration.")
                    raise ValueError(msg)

        f_ents = get_entities(img_f)
        save_run = False

        # Process event files
        if extract_events:
            _data = layout.get_events(img_f, return_type='df',
                                      derivatives=derivatives)
            if _data is not None:
                # _data = pd.read_table(evf, sep='\t')
                _data = _data.replace('n/a', np.nan)  # Replace BIDS' n/a
                _data = _data.apply(pd.to_numeric, errors='ignore')

                file_df = []

                _cols = columns or list(set(_data.columns.tolist()) -
                                        {'onset', 'duration'})

                # Construct a DataFrame for each extra column
                for col in _cols:
                    df = _data[['onset', 'duration']].copy()
                    df['condition'] = col
                    df['amplitude'] = _data[col].values
                    df['factor'] = col
                    file_df.append(df)

                # Concatenate all extracted column DFs along the row axis
                _df = pd.concat(file_df, axis=0)

                # Add in all of the event file's entities as new columns; these
                # are used for indexing later on
                for entity, value in f_ents.items():
                    _df[entity] = value

                event_dfs.append(_df)
                save_run = True

        # Process confound files
        if extract_confounds:
            sub_ents = {k: v for k, v in f_ents.items()
                        if k not in ('type', 'unique_run_id')}
            confound_files = layout.get(type='confounds', **sub_ents)
            for cf in confound_files:
                _data = pd.read_csv(cf.filename, sep='\t', na_values='n/a')
                if columns is not None:
                    conf_cols = list(set(_data.columns) & set(columns))
                    _data = _data.loc[:, conf_cols]
                if len(_data.columns) == 0:
                    continue
                confound_dfs.append(_data)
                save_run = True

        # Process recording files
        if extract_recordings:
            rec_files = layout.get_nearest(img_f, extensions='.tsv.gz',
                                           all_=True, type=['physio', 'stim'],
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

                n_rows = int(duration * freq)
                if len(values) > n_rows:
                    values = values[:n_rows, :]
                elif len(values) < n_rows:
                    pad = np.zeros((n_rows - len(values), n_cols))
                    values = np.r_[values, pad]

                # Resample to the target rate. We do this before creating the
                # final Column, because in principle, the sampling rate of
                # continuous recordings could differ across runs/subjects/etc.
                x = np.arange(n_rows)
                n_new = duration * sampling_rate
                x_new = np.linspace(0, n_rows - 1, num=n_new)
                new_vals = np.zeros((n_new, n_cols))

                for i, col_name in enumerate(rf_cols):
                    f = interp1d(x, values[:, i], kind=interp_method)
                    new_vals[:, i] = f(x_new)

                _df = pd.DataFrame(new_vals, columns=rf_cols)
                rec_dfs.append(_df)

                save_run = True

        # Track run info and add to collection
        if save_run:
            run_info = BIDSRunInfo(image_file=img_f, start=run_start_time,
                                   duration=duration, entities=f_ents)
            collection.run_infos.append(run_info)
            run_start_time += duration

        if extract_events and event_dfs:
            _df = pd.concat(event_dfs, axis=0)

            if drop_na:
                _df = _df.dropna(subset=['amplitude'])

            for name, grp in _df.groupby(['factor', 'condition']):
                data = grp.apply(pd.to_numeric, errors='ignore')
                # Subjects are often labeled numerically; ignore this
                data['subject'] = grp['subject']
                _, condition = name[0], name[1]
                col = SparseEventColumn(collection, condition, data)
                collection[condition] = col

        if extract_confounds and confound_dfs:
            _df = pd.concat(confound_dfs, axis=0).reset_index(drop=True)
            for col_name in _df.columns:
                col = DenseEventColumn(collection, col_name, _df[[col_name]])
                collection[col_name] = col

        if extract_recordings and rec_dfs:
            _df = pd.concat(rec_dfs, axis=0).reset_index(drop=True)
            for col_name in _df.columns:
                col = DenseEventColumn(collection, col_name, _df[[col_name]])
                collection[col_name] = col

    # build the index
    collection._build_dense_index()

    return collection


def _load_tsv_variables(layout, unit, entities=None, columns=None,
                        prefix_level=False, **kwargs):
    ''' Helper for scans.tsv, sessions.tsv, and participants.tsv. '''
    bids_names = {
        'run': 'scans',
        'session': 'sessions',
        'subject': 'participants'
    }

    if unit not in bids_names.keys():
        raise ValueError("unit must be one of 'run', 'session', or 'subject'.")

    type_ = bids_names[unit]

    files = layout.get(extensions='.tsv', return_type='file', type=type_,
                       **kwargs)
    dfs = []

    for f in files:

        f = layout.files[f]
        _data = pd.read_table(f.path, sep='\t')

        # Entities can be defined either within the first column of the .tsv
        # file (for entities that vary by row), or from the full file path
        # (for entities constant over all rows in the file). We extract both
        # and store them in the main DataFrame alongside other variables (as
        # they'll be extracted when the Column object is initialized anyway).
        for ent_name, ent_val in f.entities.items():
            if ent_name in BASE_ENTITIES:
                _data[ent_name] = ent_val

        # Handling is a bit more convoluted for scans.tsv, because the first
        # column contains the run filename, which we also need to parse.
        if type_ == 'scans':
            image = _data.iloc[:, 0]
            _data = _data.drop(_data.columns[0], axis=1)
            dn = dirname(f.filename)
            paths = [join(dn, p) for p in image.values]
            ent_recs = [layout.files[p].entities for p in paths
                        if p in layout.files]
            ent_cols = pd.DataFrame.from_records(ent_recs)
            _data = pd.concat([_data, ent_cols], axis=1)
            # It's possible to end up with duplicate entity columns this way
            _data = _data.T.drop_duplicates().T

        # BIDS spec requires ID columns to be named 'session_id', 'run_id',
        # etc., and IDs begin with entity prefixes (e.g., 'sub-01'). To ensure
        # consistent internal handling, we strip these suffixes and prefixes.
        elif type_ == 'sessions':
            _data = _data.rename(columns={'session_id': 'session'})
            _data['session'] = _data['session'].str.replace('ses-', '')
        elif type_ == 'participants':
            _data = _data.rename(columns={'participant_id': 'subject'})
            _data['subject'] = _data['subject'].str.replace('sub-', '')
        dfs.append(_data)

    collection = BIDSVariableCollection(unit, entities)

    if not dfs:
        warnings.warn("No %s.tsv files found in specified BIDSLayout." % type_)
        return None

    data = pd.concat(dfs, axis=0)

    ent_cols = list(set(BASE_ENTITIES) & set(data.columns))
    amp_cols = list(set(data.columns) - set(ent_cols))

    for col_name in amp_cols:

        # Rename colummns: values must be in 'amplitude'
        _data = data.loc[:, [col_name] + ent_cols]
        _data.columns = ['amplitude'] + ent_cols

        if prefix_level:
            col_name = '%s#%s' % (type_, col_name)

        col = SimpleColumn(collection, col_name, _data, unit)
        # TODO: Figure out some configurable way to handle name conflicts
        # between files. This can be quite common if, e.g., users are using
        # sessions.tsv to report the average value for rows in scans.tsv.
        # if col_name in collection.columns:
        #     raise ValueError("Name conflict: column '%s' in %s.tsv "
        #                      "already exists--probably because it's "
        #                      "defined in an events.tsv file. Please use "
        #                      "unique names." % (col_name, type_))
        collection[col_name] = col

    return collection


def load_run_variables(layout, entities=None, columns=None, **kwargs):
    return _load_tsv_variables(layout, 'run', entities, columns, **kwargs)


def load_session_variables(layout, entities=None, columns=None, **kwargs):
    return _load_tsv_variables(layout, 'session', entities, columns, **kwargs)


def load_subject_variables(layout, entities=None, columns=None, **kwargs):
    return _load_tsv_variables(layout, 'subject', entities, columns, **kwargs)


def load_variables(layout, levels=None, merge=False, target=None, **kwargs):
    ''' A convenience wrapper for one or more load_*_variables() calls.
    Args:
        layout (BIDSLayout): BIDSLayout containing variable files.
        levels (str, list): Level or list of levels to load variables for.
            Valid values are 'time', 'run', 'session', and 'subject'.
        merge (bool): If True, the requested levels are merged into a single
            BIDSVariableCollection before returning. Ignored if only one
            level is requested.
        target (str): If merge=True, target indicates the level that defines
            the granularity of the result. See merge_collections for further
            explanation.
        kwargs: Optional keyword arguments to pass onto the individual
            load_*_variables() calls.
    Returns:
        If only a single level is passed, or merge is True, a single
            BIDSVariableCollection. If a list of levels is passed and merge is
            False, a dict is returned, with level names in keys and
            BIDSVariableCollections in values.
    '''

    ALL_LEVELS = ['time', 'run', 'session', 'subject']

    if levels is None:
        levels = ALL_LEVELS

    _levels = listify(levels)

    func_map = {
        'time': load_event_variables,
        'run': load_run_variables,
        'session': load_session_variables,
        'subject': load_subject_variables
    }

    bad_levels = set(_levels) - set(ALL_LEVELS)
    if bad_levels:
        raise ValueError("Invalid level names: %s" % bad_levels)

    collections = [func_map[l](layout, **kwargs) for l in _levels]

    if len(collections) == 1:
        return collections[0]

    # if merge:
    #     return merge_collections(collections, target=target)

    return dict(zip(_levels, collections))


def merge_collections(collections):
    # For the moment, assume collections are always at same level
    collections[0].columns.update(collections[1].columns)
    return collections[0]

# def merge_collections(collections, target=None, agg_func='mean',
#                       categorical_agg_func=None, missing='fail',
#                       constant_event_values=True):

    # # Make sure the list is sorted by level
    # levels = ['time', 'run', 'session', 'subject', 'dataset']
    # collections = sorted(collections, key=lambda x: levels.index(x.level))

    # if target is None:
    #     target = collections[0].level

    # ranks = [levels.index(c.level) for c in collections]

    # # Start from the first collection and repeatedly merge with neighbor
    # # If no aggregation function is provided for categoricals, we fail if any
    # # categorical variable has more than one unique value.


class BIDSColumn(object):

    ''' Base representation of a column in a BIDS project. '''

    __metaclass__ = ABCMeta

    def __init__(self, collection, name, values):
        self.collection = collection
        self.name = name
        self.values = values

    def clone(self, data=None, **kwargs):
        ''' Clone (deep copy) the current column, optionally replacing its
        data and/or any other attributes.
        Args:
            data (DataFrame, ndarray): Optional new data to substitute into
                the cloned column. Must have same dimensionality as the
                original.
            kwargs (dict): Optional keyword arguments containing new attribute
                values to set in the copy. E.g., passing `name='my_name'`
                would set the `.name` attribute on the cloned instance to the
                passed value.
        '''
        result = deepcopy(self)
        if data is not None:
            if data.shape != self.values.shape:
                raise ValueError("Replacement data has shape %s; must have "
                                 "same shape as existing data %s." %
                                 (data.shape, self.values.shape))
            result.values = pd.Series(data)

        if kwargs:
            for k, v in kwargs.items():
                setattr(result, k, v)

        # Need to update name on Series as well
        result.values.name = kwargs.get('name', self.name)
        return result

    @abstractmethod
    def aggregate(self, unit, level, func):
        pass

    def __deepcopy__(self, memo):

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Any variables we don't want to deep copy, and should pass by ref--
        # e.g., the existing collection.
        skip_vars = ['collection']

        for k, v in self.__dict__.items():
            new_val = getattr(self, k) if k in skip_vars else deepcopy(v, memo)
            setattr(result, k, new_val)
        return result

    @abstractproperty
    def index(self):
        pass

    def get_grouper(self, groupby='unique_run_id'):
        ''' Return a pandas Grouper object suitable for use in groupby calls.
        Args:
            groupby (str, list): Name(s) of column(s) defining the grouper
                object. Anything that would be valid inside a .groupby() call
                on a pandas structure.
        Returns:
            A pandas Grouper object constructed from the specified columns
                of the current index.
        '''
        return pd.core.groupby._get_grouper(self.index, groupby)[0]

    def apply(self, func, groupby='unique_run_id', *args, **kwargs):
        ''' Applies the passed function to the groups defined by the groupby
        argument. Works identically to the standard pandas df.groupby() call.
        Args:
            func (callable): The function to apply to each group.
            groupby (str, list): Name(s) of column(s) defining the grouping.
            args, kwargs: Optional positional and keyword arguments to pass
                onto the function call.
        '''
        grouper = self.get_grouper(groupby)
        return self.values.groupby(grouper).apply(func, *args, **kwargs)


class SimpleColumn(BIDSColumn):
    ''' Represents a simple design matrix column that has no timing
    information.
    Args:
        collection (BIDSVariableCollection): The collection the current column
            is bound to and derived from.
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing a column
            named 'amplitude' as well as any identifying entities.
        factor_name (str): If this column is derived from a categorical factor
            (e.g., level 'A' in a 'trial_type' column), the name of the
            originating factor.
        level_index (int): The positional index of the current level in the
            originating categorical factor. Ignored if factor_name is None.
        level_name (str): The name of the current level in the originating
            categorical factor, if applicable.
    '''

    # Columns that define special properties (e.g., onset, duration). These
    # will be stored separately from the main data object, and are accessible
    # as properties on the SimpleColumn instance.
    _property_columns = set()
    _entity_columns = {'condition', 'amplitude', 'factor'}

    def __init__(self, collection, name, data, factor_name=None,
                 level_index=None, level_name=None):

        self.factor_name = factor_name
        self.level_index = level_index
        self.level_name = level_name

        for sc in self._property_columns:
            setattr(self, sc, data[sc].values)

        ent_cols = list(set(data.columns) - self._entity_columns -
                        self._property_columns)
        self.entities = data.loc[:, ent_cols]

        values = data['amplitude'].reset_index(drop=True)
        values.name = name

        super(SimpleColumn, self).__init__(collection, name, values)

    def aggregate(self, unit, func='mean'):

        levels = ['run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & set(self.entities.columns)
        groupby = list(groupby)

        entities = self.entities.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleColumn(self.collection, self.name, data, self.factor_name,
                            self.level_index, self.level_name)

    def to_df(self, condition=True, entities=True):
        ''' Convert to a DataFrame, with columns for onset/duration/amplitude
        plus (optionally) name and entities.
        Args:
            condition (bool): If True, adds a column for condition name, and
                names the amplitude column 'amplitude'. If False, returns just
                onset, duration, and amplitude, and gives the amplitude column
                the current column name.
            entities (bool): If True, adds extra columns for all entities.
        '''
        amp = 'amplitude' if condition else self.name
        data = pd.DataFrame({amp: self.values.values.ravel()})

        for sc in self._property_columns:
            data[sc] = getattr(self, sc)

        if condition:
            data['condition'] = self.name

        if entities:
            ent_data = self.entities.reset_index(drop=True)
            data = pd.concat([data, ent_data], axis=1)

        return data

    def split(self, grouper):
        ''' Split the current SparseEventColumn into multiple columns.
        Args:
            grouper (iterable): list to groupby, where each unique value will
                be taken as the name of the resulting column.
        Returns:
            A list of SparseEventColumns, one per unique value in the grouper.
        '''
        data = self.to_df(condition=True, entities=False)
        data = data.drop('condition', axis=1)
        # data = pd.DataFrame(dict(onset=self.onset, duration=self.duration,
        #                          amplitude=self.values.values))
        # data = pd.concat([data, self.index.reset_index(drop=True)], axis=1)

        subsets = []
        for i, (name, g) in enumerate(data.groupby(grouper)):
            name = '%s.%s' % (self.name, name)
            col = self.__class__(self.collection, name, g, level_name=name,
                                 factor_name=self.name, level_index=i)
            subsets.append(col)
        return subsets

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.entities


class SparseEventColumn(SimpleColumn):
    ''' A sparse representation of a single column of events.
    Args:
        collection (BIDSVariableCollection): The collection the current column
            is bound to and derived from.
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing the columns
            'onset', 'duration', and 'amplitude'.
        factor_name (str): If this column is derived from a categorical factor
            (e.g., level 'A' in a 'trial_type' column), the name of the
            originating factor.
        level_index (int): The positional index of the current level in the
            originating categorical factor. Ignored if factor_name is None.
        level_name (str): The name of the current level in the originating
            categorical factor, if applicable.
    '''

    _property_columns = {'onset', 'duration'}

    def to_dense(self, sampling_rate=None):
        ''' Convert the current sparse column to a dense representation.
        Returns: A DenseEventColumn. '''
        if sampling_rate is None:
            sampling_rate = self.collection.sampling_rate
        duration = int(sampling_rate * len(self.collection.dense_index) /
                       self.collection.sampling_rate)
        ts = np.zeros(duration)

        onsets = np.ceil(self.onset * sampling_rate).astype(int)
        durations = np.round(self.duration * sampling_rate).astype(int)

        for i, row in enumerate(self.values.values):
            file_id = self.entities['unique_run_id'].values[i]
            run_onset = self.collection.run_infos[file_id].start
            ev_start = onsets[i] + int(math.ceil(run_onset * sampling_rate))
            ev_end = ev_start + durations[i]
            ts[ev_start:ev_end] = row

        ts = pd.DataFrame(ts)

        return DenseEventColumn(self.collection, self.name, ts)


class DenseEventColumn(BIDSColumn):
    ''' A dense representation of a single column.

    Args:
        collection (BIDSEventVariableCollection): The collection the
            DenseEventColumn belongs to
        name (str): The name of the column
        values (NDArray): The values/amplitudes to store
        sampling_rate (float): Optional sampling rate (in Hz) to use. Must
            match the sampling rate used to generate the values. If None,
            the collection's sampling rate will be used.
    '''

    def __init__(self, collection, name, values, sampling_rate=None):
        if sampling_rate is None:
            sampling_rate = collection.sampling_rate
        self.sampling_rate = sampling_rate
        super(DenseEventColumn, self).__init__(collection, name, values)
        self._index = _build_dense_index(self.collection.run_infos,
                                         sampling_rate)

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self._index

    def split(self, grouper):
        ''' Split the current DenseEventColumn into multiple columns.
        Args:
            grouper (DataFrame): binary DF specifying the design matrix to
                use for splitting. Number of rows must match current
                DenseEventColumn; a new DenseEventColumn will be generated
                for each column in the grouper.
        Returns:
            A list of DenseEventColumns, one per unique value in the grouper.
        '''
        df = grouper * self.values
        names = df.columns
        return [DenseEventColumn(self.collection, '%s.%s' % (self.name, name),
                                 df[name].values)
                for i, name in enumerate(names)]

    def aggregate(self, unit, func='mean'):

        levels = ['run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & \
            set(self.index.columns)
        groupby = list(groupby)

        entities = self._index.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values.ravel()})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleColumn(self.collection, self.name, data)

    def resample(self, sampling_rate, kind='linear'):
        ''' Resample the column to the specified sampling rate.

        Args:
            sampling_rate (int, float): Target sampling rate (in Hz)
            kind (str): Argument to pass to scipy's interp1d; indicates the
                kind of interpolation approach to use. See interp1d docs for
                valid values.
        '''

        if sampling_rate == self.sampling_rate:
            return

        # Use the collection's index if possible
        if sampling_rate == self.collection.sampling_rate:
            self._index = self.collection.index
            return

        old_sr = self.sampling_rate
        n = len(self.index)

        self._index = _build_dense_index(self.collection.run_infos,
                                         sampling_rate)

        x = np.arange(n)
        num = int(np.ceil(n * sampling_rate / old_sr))

        f = interp1d(x, self.values.values.ravel(), kind=kind)
        x_new = np.linspace(0, n - 1, num=num)
        self.values = pd.DataFrame(f(x_new))

        self.sampling_rate = sampling_rate


BIDSRunInfo = namedtuple('BIDSRunInfo', ('image_file', 'start', 'duration',
                                         'entities'))


class BIDSVariableCollection(object):

    ''' A container for one or more variables extracted from variable files
    at a single level of analysis defined in the BIDS spec (i.e., 'run',
    'session', 'subject', or 'dataset').
    Args:
        unit (str): The unit of analysis. Each row in the stored column(s)
            is taken to reflect a single unit. Must be one of 'time', 'run',
            'session', 'subject', or 'dataset'.
        entities (list): A list of entities defined for all variables in this
            collection.
    '''

    def __init__(self, unit, entities):

        self.unit = unit
        self.entities = entities
        self.columns = {}
        self.dense_index = None

    def merge_columns(self, columns=None):
        ''' Merge columns into one DF.
        Args:
            columns (list): Optional list of column names to retain; if None,
                all columns are written out.
        Returns: A pandas DataFrame.
        '''

        _cols = self.columns.values()

        # Retain only specific columns if desired
        if columns is not None:
            _cols = [c for c in _cols if c.name in columns]

        return pd.concat([c.to_df() for c in _cols], axis=0)

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        columns are deep-cloned.
        '''
        clone = copy(self)
        clone.columns = {k: v.clone() for (k, v) in self.columns.items()}
        return clone

    def aggregate(self, unit, agg_func='mean', categorical_agg_func=None):
        ''' Aggregate variable values from a lower level at a higher level.
        Args:
            unit (str): The unit of aggregation. The returned collection will
                have one row per value of this unit.
            agg_func (str, Callable): Aggregation function to use. Must be
                either a named function recognized by apply() in pandas, or a
                Callable that takes a DataFrame and returns a Series or
                DataFrame.
            categorical_agg_func (str, Callable): Aggregation function to use
                for categorical variables. Must be a function that returns
                valid output given categorical inputs. If None, aggregation
                will only proceed if all categorical columns have exactly one
                unique value.
        '''

        for col in self.columns.values():
            if is_numeric_dtype(col.values):
                _func = agg_func
            else:
                if categorical_agg_func is not None:
                    _func = categorical_agg_func
                elif col.values.nunique() > 1:
                    msg = ("Column %s is categorical and has more than one "
                           "unique value. You must explicitly specify an "
                           "aggregation function in the categorical_agg_func "
                           "argument.")
                    raise ValueError(msg)
                else:
                    _func = 'first'

            self[col.name] = col.aggregate(unit, _func)

    def __getitem__(self, col):
        return self.columns[col]

    def __setitem__(self, col, obj):
        # Ensure name matches collection key, but raise warning if needed.
        if obj.name != col:
            warnings.warn("The provided key to use in the collection ('%s') "
                          "does not match the passed Column object's existing "
                          "name ('%s'). The Column name will be set to match "
                          "the provided key." % (col, obj.name))
            obj.name = col
        self.columns[col] = obj

    def match_columns(self, pattern, return_type='name'):
        ''' Return columns whose names match the provided regex pattern.
        Args:
            pattern (str): A regex pattern to match all column names against.
            return_type (str): What to return. Must be one of:
                'name': Returns a list of names of matching columns.
                'column': Returns a list of Column objects whose names match.
        '''
        pattern = re.compile(pattern)
        cols = [c for c in self.columns.values() if pattern.search(c.name)]
        return cols if return_type.startswith('col') \
            else [c.name for c in cols]

    def _construct_design_matrix(self, data, groupby=None, aggregate=None,
                                 add_intercept=None, drop_entities=False,
                                 drop_cols=None, **kwargs):

        # subset the data if needed
        if kwargs:
            # TODO: make sure this handles constraints on int columns properly
            bad_keys = list(set(kwargs.keys()) - set(data.columns))
            if bad_keys:
                raise ValueError("The following query constraints do not map "
                                 "onto existing columns: %s." % bad_keys)
            query = ' and '.join(["{} in {}".format(k, v)
                                  for k, v in kwargs.items()])
            data = data.query(query)

        if groupby and aggregate is not None:
            groupby = list(set(groupby) & set(data.columns))
            data = data.groupby(groupby).agg(aggregate).reset_index()

        if add_intercept:
            data.insert(0, 'intercept', 1)

        if drop_cols is None:
            drop_cols = []

        # Always drop columns meant for internal use
        drop_cols += ['unique_run_id', 'time']

        # Optionally drop entities
        if drop_entities:
            drop_cols += self.entities

        drop_cols = list(set(drop_cols) & set(data.columns))
        data = data.drop(drop_cols, axis=1).reset_index(drop=True)
        return data

    def get_design_matrix(self, columns=None, groupby=None, aggregate=None,
                          add_intercept=False, drop_entities=False, **kwargs):
        ''' Returns a design matrix constructed by combining the current
        BIDSVariableCollection's columns.
        Args:
            columns (list): Optional list naming columns to include in the
                design matrix. If None (default), all columns are included.
            groupby (str, list): Optional name (or list of names) of design
                variables to group by.
            aggregate (str, Callable): Optional aggregation function to apply
                to groups if groupby is not None (ignored otherwise). Must be
                either a named function recognized by pandas, or a Callable
                that takes a DataFrame and returns a Series or DataFrame.
            add_intercept (bool): If True, adds an intercept column (i.e.,
                constant column of 1's) to the returned design matrix.
            drop_entities (bool): If True, entities are stripped from the
                returned design matrix. When False, entities like 'subject' and
                'run' are included as columns in the matrix, leaving it up to
                the user to remove them if appropriate.
            kwargs: Optional query constraints understood by
                pd.DataFrame.query(). Must be compatible with the 'in' syntax;
                e.g., passing subject=['01', '02', '03'] would return only
                rows that match the first 3 subjects.

        Returns: A pandas DataFrame, where units of analysis (seconds, runs,
            etc.) are in rows, and variables/columns are in columns.

        '''

        if columns is None:
            columns = list(self.columns.keys())

        if groupby is None:
            groupby = []

        data = self.merge_columns(columns=columns)

        return self._construct_design_matrix(data, groupby, aggregate,
                                             add_intercept, drop_entities,
                                             **kwargs)


class BIDSEventVariableCollection(BIDSVariableCollection):

    ''' A container for one or more EventColumns--i.e., Columns that have a
    temporal dimension.

    Args:
        unit (str): The unit of analysis. Each row in the stored column(s)
            is taken to reflect a single unit. Must be one of 'time', 'run',
            'session', 'subject', or 'dataset'.
        entities (list): A list of entities defined for all variables in this
            collection.
        sampling_rate (float): Sampling rate (in Hz) to use when working with
            dense representations of variables.
        repetition_time (float): TR of corresponding image(s) in seconds.
    '''

    def __init__(self, unit, entities, sampling_rate=None,
                 repetition_time=None):

        self.sampling_rate = sampling_rate
        self.repetition_time = repetition_time
        self.run_infos = []
        self.dense_index = None
        super(BIDSEventVariableCollection, self).__init__(unit, entities)

    def _get_sampling_rate(self, sr):
        return self.repetition_time if sr == 'tr' else sr

    def _build_dense_index(self):
        ''' Build an index of all tracked entities for all dense columns. '''

        if not self.run_infos:
            return

        self.dense_index = _build_dense_index(self.run_infos,
                                              self.sampling_rate)

    def _none_dense(self):
        return all([isinstance(c, SimpleColumn)
                    for c in self.columns.values()])

    def _all_dense(self):
        return all([isinstance(c, DenseEventColumn)
                    for c in self.columns.values()])

    @property
    def index(self):
        return self.dense_index

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        columns and the index are deep-cloned.
        '''
        clone = super(BIDSEventVariableCollection, self).clone()
        clone.dense_index = self.dense_index.copy()
        return clone

    def resample(self, sampling_rate='tr', force_dense=False, in_place=False,
                 kind='linear'):
        ''' Resample all dense columns (and optionally, sparse ones) to the
        specified sampling rate.

        Args:
            sampling_rate (int, float): Target sampling rate (in Hz)
            force_dense (bool): if True, all sparse columns will be forced to
                dense.
            in_place (bool): When True, all columns are overwritten in-place.
                When False, returns resampled versions of all columns.
            kind (str): Argument to pass to scipy's interp1d; indicates the
                kind of interpolation approach to use. See interp1d docs for
                valid values.
        '''

        # Store old sampling rate-based variables
        sampling_rate = self._get_sampling_rate(sampling_rate)

        columns = {}

        for name, col in self.columns.items():
            if isinstance(col, SparseEventColumn):
                if force_dense and is_numeric_dtype(col.values):
                    columns[name] = col.to_dense(sampling_rate)
            else:
                col = col.clone()
                col.resample(sampling_rate, kind)
                columns[name] = col

        if in_place:
            for k, v in columns.items():
                self.columns[k] = v
            # Rebuild the dense index
            self.sampling_rate = sampling_rate
            self._build_dense_index()

        else:
            return columns

    def merge_columns(self, columns=None, sparse=True, sampling_rate='tr'):
        ''' Merge columns into one DF.
        Args:
            columns (list): Optional list of column names to retain; if None,
                all columns are written out.
            sparse (bool): If True, columns will be kept in a sparse format
                provided they are all internally represented as such. If False,
                a dense matrix (i.e., uniform sampling rate for all events)
                will be exported. Will be ignored if at least one column is
                dense.
            sampling_rate (float): If a dense matrix is written out, the
                sampling rate (in Hz) to use for downsampling. Defaults to the
                value currently set in the instance.
        Returns: A pandas DataFrame.
        '''

        if sparse and self._none_dense():
            return super(BIDSEventVariableCollection,
                         self).merge_columns(columns)

        sampling_rate = self._get_sampling_rate(sampling_rate)

        # Make sure all columns have the same sampling rate
        _cols = self.resample(sampling_rate, force_dense=True,
                              in_place=False).values()

        # Retain only specific columns if desired
        if columns is not None:
            _cols = [c for c in _cols if c.name in columns]

        _cols = [c for c in _cols if c.name not in ["event_file_id", "time"]]

        # Merge all data into one DF
        dfs = [pd.Series(c.values.iloc[:, 0], name=c.name) for c in _cols]
        # Convert datetime to seconds and add duration column
        dense_index = self.dense_index.copy()
        onsets = self.dense_index.pop('time').values.astype(float) / 1e+9
        timing = pd.DataFrame({'onset': onsets})
        timing['duration'] = 1. / sampling_rate
        dfs = [timing] + dfs + [dense_index]
        data = pd.concat(dfs, axis=1)

        return data

    def get_design_matrix(self, columns=None, groupby=None, aggregate=None,
                          add_intercept=False, drop_entities=False,
                          drop_timing=False, sampling_rate='tr', **kwargs):
        ''' Returns a design matrix constructed by combining the current
        BIDSVariableCollection's columns.
        Args:
            columns (list): Optional list naming columns to include in the
                design matrix. If None (default), all columns are included.
            groupby (str, list): Optional name (or list of names) of design
                variables to group by.
            aggregate (str, Callable): Optional aggregation function to apply
                to groups if groupby is not None (ignored otherwise). Must be
                either a named function recognized by pandas, or a Callable
                that takes a DataFrame and returns a Series or DataFrame.
            add_intercept (bool): If True, adds an intercept column (i.e.,
                constant column of 1's) to the returned design matrix.
            drop_entities (bool): If True, entities are stripped from the
                returned design matrix. When False, entities like 'subject' and
                'run' are included as columns in the matrix, leaving it up to
                the user to remove them if appropriate.
            sampling_rate (float, str): The sampling rate to use for the
                returned design matrix. Value must be either a float
                expressed in seconds, or the special value 'tr', which
                uses the associated scan's repetition time as the sampling
                rate.
            kwargs: Optional query constraints understood by
                pd.DataFrame.query(). Must be compatible with the 'in' syntax;
                e.g., passing subject=['01', '02', '03'] would return only
                rows that match the first 3 subjects.

        Returns: A pandas DataFrame, where units of analysis (seconds, runs,
            etc.) are in rows, and variables/columns are in columns.

        '''
        if columns is None:
            columns = list(self.columns.keys())

        if groupby is None:
            groupby = []

        data = self.merge_columns(columns=columns, sampling_rate=sampling_rate,
                                  sparse=True)

        return self._construct_design_matrix(data, groupby, aggregate,
                                             add_intercept, drop_entities,
                                             **kwargs)


def _build_dense_index(run_infos, sampling_rate):
    ''' Build an index of all tracked entities for all dense columns. '''
    index = []
    sr = int(1000. / sampling_rate)
    for run in run_infos:
        reps = int(math.ceil(run.duration * sampling_rate))
        ent_vals = list(run.entities.values())
        data = np.broadcast_to(ent_vals, (reps, len(ent_vals)))
        df = pd.DataFrame(data, columns=list(run.entities.keys()))
        df['time'] = pd.date_range(0, periods=len(df), freq='%sms' % sr)
        index.append(df)
    return pd.concat(index, axis=0).reset_index(drop=True)
