import numpy as np
import pandas as pd
from bids.grabbids import BIDSLayout
import nibabel as nb
import warnings
from collections import namedtuple
import math
from copy import copy, deepcopy
from abc import abstractproperty, ABCMeta
from bids.utils import listify
import re
from scipy.interpolate import interp1d
from pandas.api.types import is_numeric_dtype


BASE_ENTITIES = ['subject', 'session', 'task', 'run']


def load_variables(layouts, default_duration=0, entities=None, columns=None,
                   sampling_rate=10, drop_na=True, extra_paths=None,
                   overwrite_bids_variables=False, **selectors):
    ''' Loads all variables found in one or more BIDSLayouts and returns them
    as a BIDSVariableCollection.

    Args:
        layouts (str, list): Path(s) to the root of the BIDS project(s). Can
            also be a list of Layouts.
        default_duration (float): Default duration to assign to events in cases
            where a duration is missing.
        entities (list): Optional list of entities to encode and index by.
            If None (default), uses the standard BIDS-defined entities of
            ['run', 'session', 'subject', 'task'].
        columns (list): Optional list of names specifying which columns in the
            event files to read. By default, reads all columns found.
        sampling_rate (int, float): Sampling rate to use internally when
            converting sparse columns to dense format.
        drop_na (bool): If True, removes all events where amplitude is n/a. If
            False, leaves n/a values intact. Note that in the latter case,
            transformations that requires numeric values may fail.
        extra_paths (None, list): Optional list of additional paths to
            files or folders containing design information to read from.
            If None (default), will use all event files found within the
            current BIDS project.
        overwrite_bids_variables (bool): If `extra_paths` is passed and
            this argument is True, existing event files within the BIDS
            project are completely ignored.
        selectors (dict): Optional keyword arguments passed onto the
            BIDSLayout instance's get() method; can be used to constrain
            which data are loaded.

    Notes:
        If extra_paths is passed, the following two constraints must apply
        to every detected file:
            * ALL relevant BIDS entities MUST be encoded in each file.
              I.e., "sub-02_task-mixedgamblestask_run-01_events.tsv" is
              valid, but "subject_events.tsv" would not be.
            * It is assumed that all and only the event files in the folder
              are to be processed--i.e., there cannot be any other
              subjects, runs, etc. whose events also need to be processed
              but that are missing from the passed list.
    '''

    # Load Layouts and merge them
    layouts = listify(layouts)
    layouts = [BIDSLayout(l) if isinstance(l, str) else l for l in layouts]
    if len(layouts) > 1:
        for l in layouts[1:]:
            layouts[0].files.update(l.files)

            for k, v in l.entities.items():
                if k not in layouts[0].entities:
                    layouts[0].entities.update(v)
                else:
                    layouts[0].entities[k].files.update(v.files)

    layout = layouts[0]
    entities = entities if entities is not None else BASE_ENTITIES
    event_files = []

    # Loop over images and get all corresponding event files
    images = layout.get(return_type='file', type='bold', modality='func',
                        extensions='.nii.gz', **selectors)

    if not images:
        raise ValueError("No functional images that match criteria found.")

    event_files = []
    run_trs = []

    for img_f in images:

        evf = layout.get_events(img_f)
        if not evf:
            raise ValueError("Could not find event file that matches %s." %
                             img_f)
        tr = 1. / layout.get_metadata(img_f)['RepetitionTime']
        run_trs.append(tr)

        # Because of inheritance, _events.tsv filenames don't always
        # contain all entities, so we first get them from the image
        # file, then update with anything found in the event file.
        f_ents = layout.files[img_f].entities.copy()
        f_ents.update(layout.files[evf].entities)
        f_ents = {k: v for k, v in f_ents.items() if k in entities}
        event_files.append((evf, img_f, f_ents))

    if len(set(run_trs)) > 1:
        raise ValueError("More than one TR detected across specified runs."
                         " Currently we can only handle runs with the same"
                         " repetition time.")

    repetition_time = run_trs[0]

    dfs = []
    start_time = 0

    collection = BIDSVariableCollection(layouts[0], entities=entities,
                                        default_duration=default_duration,
                                        sampling_rate=sampling_rate,
                                        repetition_time=repetition_time)

    ### HANDLE ALL BIDS VARIABLE FILES ###

    # Load events.tsv
    for (evf, img_f, f_ents) in event_files:
        _data = pd.read_table(evf, sep='\t')
        _data = _data.replace('n/a', np.nan)  # Replace BIDS' n/a
        _data = _data.apply(pd.to_numeric, errors='ignore')

        # Get duration of run: first try to get it directly from the image
        # header; if that fails, fall back on taking the offset of the last
        # event in the event file--but raise warning, because this is
        # suboptimal (e.g., there could be empty volumes at the end).
        try:
            img = nb.load(img_f)
            duration = img.shape[3] * img.header.get_zooms()[-1]
        except Exception as e:
            duration = (_data['onset'] + _data['duration']).max()
            warnings.warn("Unable to extract scan duration from one or more"
                          "images; setting duration to the offset of the last "
                          "detected event instead--but note that this may "
                          "produce unexpected results.")

        # Add default values for entities that may not be passed explicitly
        evf_index = len(collection.event_files)
        f_ents['event_file_id'] = evf_index

        ef = BIDSEventFile(event_file=evf, image_file=img_f,
                           start=start_time, duration=duration,
                           entities=f_ents)
        collection.event_files.append(ef)
        start_time += duration

        skip_cols = ['onset', 'duration']

        file_df = []

        _columns = columns

        # By default, read all columns from the event file
        if _columns is None:
            _columns = list(set(_data.columns.tolist()) - set(skip_cols))

        # Construct a DataFrame for each extra column
        for col in _columns:
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

        dfs.append(_df)

    _df = pd.concat(dfs, axis=0)

    if drop_na:
        _df = _df.dropna(subset=['amplitude'])

    for name, grp in _df.groupby(['factor', 'condition']):
        data = grp.apply(pd.to_numeric, errors='ignore')
        _, condition = name[0], name[1]
        collection[condition] = SparseEventColumn(collection, condition, data,
                                                  block_level='run')

    # sessions.tsv and participants.tsv have the same format and differ only
    # in level assignment.
    variable_files = [('sessions', 'session'), ('participants', 'subject')]

    for type_, level in variable_files:

        files = layout.get(extensions='.tsv', type=type_, **selectors)
        dfs = [pd.read_table(f.filename, sep='\t') for f in files]

        if not dfs:
            continue

        data = pd.concat(dfs, axis=0)

        for i, col_name in enumerate(data.columns[1:]):
            # Rename colummns: values must be in 'amplitude', and users
            # sometimes give the ID column the wrong name.
            old_lev_name = data.columns[i]
            _data = data.loc[:, [old_lev_name, col_name]]
            _data.columns = [level, 'amplitude']
            col = SimpleColumn(collection, col_name, _data, level)
            if col_name in collection.columns:
                raise ValueError("Name conflict: column '%s' in %s.tsv "
                                 "already exists--probably because it's "
                                 "defined in an events.tsv file. Please use "
                                 "unique names." % (col_name, level))
            collection[col_name] = col

    # build the index
    collection._build_dense_index()

    return collection


class BIDSColumn(object):

    ''' Base representation of a column in a BIDS project. '''

    __metaclass__ = ABCMeta

    def __init__(self, collection, name, values, block_level=None):
        self.collection = collection
        self.name = name
        self.values = values
        self.block_level = block_level

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

    def get_grouper(self, groupby='event_file_id'):
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

    def apply(self, func, groupby='event_file_id', *args, **kwargs):
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

    '''

    # Columns that define special properties (e.g., onset, duration). These
    # will be stored separately from the main data object, and are accessible
    # as properties on the SimpleColumn instance.
    _property_columns = set()
    _entity_columns = {'condition', 'amplitude', 'factor'}

    def __init__(self, collection, name, data, block_level=None,
                 factor_name=None, level_index=None, level_name=None):

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

        super(SimpleColumn, self).__init__(collection, name, values,
                                           block_level)

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
            name = '%s/%s' % (self.name, name)
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

    def to_dense(self):
        ''' Convert the current sparse column to a dense representation. '''
        sampling_rate = self.collection.sampling_rate
        duration = len(self.collection.dense_index)
        ts = np.zeros(duration)

        onsets = np.ceil(self.onset * sampling_rate).astype(int)
        durations = np.round(self.duration * sampling_rate).astype(int)

        for i, row in enumerate(self.values.values):
            file_id = self.entities['event_file_id'].values[i]
            run_onset = self.collection.event_files[file_id].start
            ev_start = onsets[i] + int(math.ceil(run_onset * sampling_rate))
            ev_end = ev_start + durations[i]
            ts[ev_start:ev_end] = row

        ts = pd.DataFrame(ts)

        return DenseEventColumn(self.collection, self.name, ts)


class DenseEventColumn(BIDSColumn):
    ''' A dense representation of a single column. '''

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.collection.dense_index

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
        return [DenseEventColumn(self.collection, '%s/%s' % (self.name, name),
                                 df[name].values)
                for i, name in enumerate(names)]


BIDSEventFile = namedtuple('BIDSEventFile', ('image_file', 'event_file',
                                             'start', 'duration', 'entities'))


class BIDSVariableCollection(object):

    def __init__(self, layout, entities, default_duration=None,
                 sampling_rate=None, repetition_time=None):
        self.layout = layout
        self.entities = entities
        self.default_duration = default_duration
        self.sampling_rate = sampling_rate
        self.repetition_time = repetition_time
        self.columns = {}
        self.event_files = []
        self.dense_index = None

    def get_sampling_rate(self, sr):
        return self.repetition_time if sr == 'tr' else sr

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
        # # Can only write sparse output if all columns are sparse
        force_dense = True if not sparse or not self._none_dense() else False

        sampling_rate = self.get_sampling_rate(sampling_rate)

        if force_dense and not self._all_dense():
            _cols = self.resample(sampling_rate, force_dense=force_dense,
                                  in_place=False).values()
        else:
            _cols = self.columns.values()

        # Retain only specific columns if desired
        if columns is not None:
            _cols = [c for c in _cols if c.name in columns]

        _cols = [c for c in _cols if c.name not in ["event_file_id", "time"]]

        # Merge all data into one DF
        if force_dense:
            dfs = [pd.Series(c.values.iloc[:, 0], name=c.name) for c in _cols]
            # Convert datetime to seconds and add duration column
            dense_index = self.dense_index.copy()
            onsets = self.dense_index.pop('time').values.astype(float) / 1e+9
            timing = pd.DataFrame({'onset': onsets})
            timing['duration'] = 1. / sampling_rate
            dfs = [timing] + dfs + [dense_index]
            data = pd.concat(dfs, axis=1)
        else:
            data = pd.concat([c.to_df() for c in _cols], axis=0)

        return data

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        columns and indexes are deep-cloned.
        '''
        clone = copy(self)
        clone.columns = {k: v.clone() for (k, v) in self.columns.items()}
        if clone.dense_index is not None:
            clone.dense_index = self.dense_index.copy()
        return clone

    def _none_dense(self):
        return all([isinstance(c, SimpleColumn)
                    for c in self.columns.values()])

    def _all_dense(self):
        return all([isinstance(c, DenseEventColumn)
                    for c in self.columns.values()])

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

    def _build_dense_index(self):
        ''' Build an index of all tracked entities for all dense columns. '''
        index = []
        sr = int(1000. / self.sampling_rate)
        for evf in self.event_files:
            reps = int(math.ceil(evf.duration * self.sampling_rate))
            ent_vals = list(evf.entities.values())
            data = np.broadcast_to(ent_vals, (reps, len(ent_vals)))
            df = pd.DataFrame(data, columns=list(evf.entities.keys()))
            df['time'] = pd.date_range(0, periods=len(df), freq='%sms' % sr)
            index.append(df)
        self.dense_index = pd.concat(index, axis=0).reset_index(drop=True)

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

    def get_design_matrix(self, groupby=None, results=None, columns=None,
                          aggregate=None, block_level='run',
                          add_intercept=False, sampling_rate='tr',
                          drop_entities=False, drop_timing=False, **kwargs):

        if columns is None:
            columns = list(self.columns.keys())

        if groupby is None:
            groupby = []

        if block_level is not None:
            columns = [c for c in columns
                       if self.columns[c].block_level == block_level]

        data = self.merge_columns(columns=columns, sampling_rate=sampling_rate,
                                  sparse=True)

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

        # Always drop columns meant for internal use
        drop_cols = ['event_file_id', 'time']

        if drop_timing:
            drop_cols += ['onset', 'duration']

        # Optionally drop entities
        if drop_entities:
            drop_cols += self.entities

        drop_cols = list(set(drop_cols) & set(data.columns))

        return data.drop(drop_cols, axis=1)

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

        # TODO: make this more robust; should not replace sampling_rate in
        # self until everything works successfully--but this will require
        # some refactoring.

        # Store old sampling rate-based variables
        sampling_rate = self.get_sampling_rate(sampling_rate)

        old_sr = self.sampling_rate
        n = len(self.dense_index)

        # Rebuild the dense index
        self.sampling_rate = sampling_rate
        self._build_dense_index()

        x = np.arange(n)
        num = int(np.ceil(n * sampling_rate / old_sr))

        columns = {}

        for name, col in self.columns.items():
            if isinstance(col, SparseEventColumn):
                if force_dense and is_numeric_dtype(col.values):
                    columns[name] = col.to_dense()
            else:
                col = col.clone()
                f = interp1d(x, col.values.values.ravel(), kind=kind)
                x_new = np.linspace(0, n - 1, num=num)
                col.values = pd.DataFrame(f(x_new))
                columns[name] = col

        if in_place:
            for k, v in columns.items():
                self.columns[k] = v
        else:
            return columns
