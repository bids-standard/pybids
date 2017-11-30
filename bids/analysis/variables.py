import numpy as np
import pandas as pd
from bids.grabbids import BIDSLayout
import nibabel as nb
import warnings
from collections import namedtuple
import math
from copy import copy, deepcopy
from abc import abstractproperty, ABCMeta
from six import string_types
import os
from bids.utils import listify
import json
import re
from scipy.interpolate import interp1d
from functools import partial
from pandas.api.types import is_numeric_dtype


class BIDSColumn(object):

    ''' Base representation of a column in a BIDS project. '''

    __metaclass__ = ABCMeta

    def __init__(self, manager, name, values):
        self.manager = manager
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

    def __deepcopy__(self, memo):

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Any variables we don't want to deep copy, and should pass by ref--
        # e.g., the existing manager.
        skip_vars = ['manager']

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


class SparseBIDSColumn(BIDSColumn):
    ''' A sparse representation of a single column of events.
    Args:
        manager (BIDSVariableManager): The manager the current column
            is bound to and derived from.
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing the columns
            'onset', 'duration', and 'amplitude'.
        factor_name (str): If this column is derived from a categorical factor
            (e.g., level 'A' in a 'trial_type' column), the name of the
            originating factor.
        factor_index (int): The positional index of the current level in the
            originating categorical factor. Ignored if factor_name is None.
        level_name (str): The name of the current level in the originating
            categorical factor, if applicable.
    '''

    def __init__(self, manager, name, data, factor_name=None,
                 factor_index=None, level_name=None):

        self.onsets = data['onset'].values
        self.durations = data['duration'].values
        self.factor_name = factor_name
        self.factor_index = factor_index
        self.level_name = level_name

        ent_cols = list(set(data.columns) - {'onset', 'duration', 'condition',
                                             'amplitude', 'factor'})
        self.entities = data.loc[:, ent_cols]

        values = data['amplitude'].reset_index(drop=True)
        values.name = name

        super(SparseBIDSColumn, self).__init__(manager, name, values)

    def to_dense(self):
        ''' Convert the current sparse column to a dense representation. '''
        sampling_rate = self.manager.sampling_rate
        duration = len(self.manager.dense_index)
        ts = np.zeros(duration)

        onsets = np.ceil(self.onsets * sampling_rate).astype(int)
        durations = np.round(self.durations * sampling_rate).astype(int)

        for i, row in enumerate(self.values.values):
            file_id = self.entities['event_file_id'].values[i]
            run_onset = self.manager.event_files[file_id].start
            ev_start = onsets[i] + int(math.ceil(run_onset * sampling_rate))
            ev_end = ev_start + durations[i]
            ts[ev_start:ev_end] = row

        ts = pd.DataFrame(ts)

        return DenseBIDSColumn(self.manager, self.name, ts)

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
        data = pd.DataFrame({'onset': self.onsets,
                             'duration': self.durations,
                             amp: self.values.values.ravel()})
        if condition:
            data['condition'] = self.name
        if entities:
            ent_data = self.entities.reset_index(drop=True)
            data = pd.concat([data, ent_data], axis=1)
        return data

    def split(self, grouper):
        ''' Split the current SparseBIDSColumn into multiple columns.
        Args:
            grouper (iterable): list to groupby, where each unique value will
                be taken as the name of the resulting column.
        Returns:
            A list of SparseBIDSColumns, one per unique value in the grouper.
        '''
        data = pd.DataFrame(dict(onset=self.onsets, duration=self.durations,
                                 amplitude=self.values.values))
        data = pd.concat([data, self.entities.reset_index(drop=True)], axis=1)

        subsets = []
        for i, (name, g) in enumerate(data.groupby(grouper)):
            name = '%s/%s' % (self.name, name)
            col = SparseBIDSColumn(self.manager, name, g, level_name=name,
                                   factor_name=self.name, factor_index=i)
            subsets.append(col)
        return subsets

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.entities


class DenseBIDSColumn(BIDSColumn):
    ''' A dense representation of a single column. '''

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.manager.dense_index

    def split(self, grouper):
        ''' Split the current DenseBIDSColumn into multiple columns.
        Args:
            grouper (DataFrame): binary DF specifying the design matrix to
                use for splitting. Number of rows must match current
                DenseBIDSColumn; a new DenseBIDSColumn will be generated
                for each column in the grouper.
        Returns:
            A list of DenseBIDSColumns, one per unique value in the grouper.
        '''
        df = grouper * self.values
        names = df.columns
        return [DenseBIDSColumn(self.manager, '%s/%s' % (self.name, name),
                                df[name].values)
                for i, name in enumerate(names)]


BIDSEventFile = namedtuple('BIDSEventFile', ('image_file', 'event_file',
                                             'start', 'duration', 'entities'))


class BIDSVariableManager(object):

    ''' A container for all design-related informationextracted from a BIDS
    project.
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

    def __init__(self, layouts, default_duration=0, entities=None,
                 columns=None, sampling_rate=10, drop_na=True,
                 extra_paths=None, overwrite_bids_variables=False,
                 **selectors):

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
        self.layout = layouts[0]

        if entities is None:
            entities = ['subject', 'session', 'task', 'run']
        self.entities = entities
        self.default_duration = default_duration
        self.select_columns = columns
        self.sampling_rate = sampling_rate
        self.drop_na = drop_na
        self.current_grouper = 'event_file_id'
        self.extra_paths = extra_paths
        self.overwrite_bids_variables = overwrite_bids_variables
        self.selectors = selectors

    def get_sampling_rate(self, sr):
        return self.repetition_time if sr == 'tr' else sr

    def load(self):
        ''' Read in and process event files.
        Args:
            reset (bool): If True (default), clears all previously processed
                event files and columns; if False, adds new files
                incrementally.
        '''
        self.event_files = []
        self.columns = {}

        # Loop over images and get all corresponding event files
        images = self.layout.get(return_type='file', type='bold',
                                 modality='func', extensions='.nii.gz',
                                 **self.selectors)
        if not images:
            raise ValueError("No functional images that match criteria found.")

        event_files = []
        run_trs = []

        for img_f in images:
            evf = self.layout.get_events(img_f)
            if not evf:
                raise ValueError("Could not find event file that matches %s." %
                                 img_f)
            tr = 1./self.layout.get_metadata(img_f)['RepetitionTime']
            run_trs.append(tr)

            f_ents = self.layout.files[evf].entities
            f_ents = {k: v for k, v in f_ents.items() if k in self.entities}

            event_files.append((evf, img_f, f_ents))

        if len(set(run_trs)) > 1:
            raise ValueError("More than one TR detected across specified runs."
                             " Currently we can only handle runs with the same"
                             " repetition time.")

        self.repetition_time = run_trs[0]

        dfs = []
        start_time = 0

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
                warnings.warn("Unable to extract scan duration from image %s; "
                              "setting duration to the offset of the last "
                              "detected event instead (%d)--but note that this"
                              " may produce unexpected results." %
                              (img_f, duration))

            # Add default values for entities that may not be passed explicitly
            evf_index = len(self.event_files)
            f_ents['event_file_id'] = evf_index

            ef = BIDSEventFile(event_file=evf, image_file=img_f,
                               start=start_time, duration=duration,
                               entities=f_ents)
            self.event_files.append(ef)
            start_time += duration

            skip_cols = ['onset', 'duration']

            file_df = []

            _columns = self.select_columns

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

        if self.drop_na:
            _df = _df.dropna(subset=['amplitude'])

        for name, grp in _df.groupby(['factor', 'condition']):
            data = grp.apply(pd.to_numeric, errors='ignore')
            _, condition = name[0], name[1]
            self.columns[condition] = SparseBIDSColumn(self, condition, data)

        # build the index
        self._build_dense_index()

    def merge_columns(self, columns=None, sparse=True, sampling_rate='tr'):
        ''' Merge columns into one DF.
        Args:
            columns (list): Optional list of column names to retain; if None,
                all columns are written out.
            sparse (bool): If True, events will be written out in sparse format
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
        force_dense = True if not sparse or not self._all_sparse() else False

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

    def write(self, path=None, file=None, columns=None, sparse=True,
              sampling_rate='tr', suffix='_events', header=True,
              overwrite=False):
        ''' Write out all events in manager to TSV file(s).
        Args:
            path (str): The directory to write event files to
            file (str): Event file to write events to
            suffix (str): Suffix to append to filenames
            header (bool): If True, includes column names in the header row. If
                False, omits the header row.
            overwrite (bool): If True, any existing .tsv file at the output
                location will be overwritten.
        '''

        if path is None and file is None:
            raise ValueError("Either the 'path' or the 'file' arguments must "
                             "be provided.")

        data = self.merge_columns(columns, sparse, sampling_rate)

        # By default drop columns for internal use
        _drop_cols = [c for c in data.columns
                      if c in ['event_file_id', 'time']]

        # If output is a single file, just write out the entire DF, adding in
        # the entities.
        if file is not None:
            data.drop(_drop_cols, axis=1).to_csv(file, sep='\t', header=header,
                                                 index=False)

        # Otherwise we write out one event file per entity combination
        else:
            common_ents = [e for e in self.entities if e in data.columns]
            groups = data.groupby(common_ents)
            _drop_cols += common_ents
            common_ents = ["sub" if e is "subject" else e for e in common_ents]

            for name, g in groups:
                # build file name
                filename = '_'.join(['%s-%s' % (e, name[i]) for i, e in
                                     enumerate(common_ents)])
                filename = os.path.join(path, filename + suffix + ".tsv")
                g.drop(_drop_cols, axis=1).to_csv(filename, sep='\t',
                                                  header=header, index=False)

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        columns and indexes are deep-cloned.
        '''
        clone = copy(self)
        clone.columns = {k: v.clone() for (k, v) in self.columns.items()}
        clone.dense_index = self.dense_index.copy()
        return clone

    def _all_sparse(self):
        return all([isinstance(c, SparseBIDSColumn)
                    for c in self.columns.values()])

    def _all_dense(self):
        return all([isinstance(c, DenseBIDSColumn)
                    for c in self.columns.values()])

    def __getitem__(self, col):
        return self.columns[col]

    def __setitem__(self, col, obj):
        # Ensure name matches manager key, but raise warning if needed.
        if obj.name != col:
            warnings.warn("The provided key to use in the manager ('%s') "
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
                          aggregate=None, add_intercept=True,
                          sampling_rate='tr', drop_entities=False, **kwargs):

        if columns is None:
            columns = list(self.columns.keys())

        if groupby is None:
            groupby = []

        # data = pd.concat([self.columns[c].to_df() for c in columns], axis=0)
        data = self.merge_columns(columns=columns, sampling_rate=sampling_rate,
                                  sparse=False)

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
        drop_cols = ['onset', 'duration', 'event_file_id', 'time']
        # Optionally drop entities
        if drop_entities:
            drop_cols += self.entities

        drop_cols = list(set(drop_cols) & set(data.columns))

        return data.drop(drop_cols, axis=1)

    def set_analysis_level(self, level, hierarchical=True):
        if level == 'dataset':
            level = None
        elif hierarchical:
            hierarchy = ['run', 'session', 'subject'][::-1]
            pos = hierarchy.index(level)
            level = hierarchy[:(pos + 1)]
        self.current_grouper = level

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
            if isinstance(col, SparseBIDSColumn):
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
