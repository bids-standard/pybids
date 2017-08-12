import pandas as pd
import numpy as np
from bids.grabbids import BIDSLayout
import nibabel as nb
import warnings
from collections import namedtuple
import math
from copy import deepcopy
from abc import abstractproperty, ABCMeta
from six import string_types
import os
import json
from bids.events import transform
from scipy.interpolate import interp1d


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


class SparseBIDSColumn(BIDSColumn):
    ''' A sparse representation of a single column of events.
    Args:
        collection (BIDSEventCollection): The collection the current column
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

    def __init__(self, collection, name, data, factor_name=None,
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

        super(SparseBIDSColumn, self).__init__(collection, name, values)

    def to_dense(self):
        ''' Convert the current sparse column to a dense representation. '''

        sampling_rate = self.collection.sampling_rate
        duration = len(self.collection.dense_index)
        ts = np.zeros(duration)

        onsets = np.ceil(self.onsets * sampling_rate).astype(int)
        durations = np.round(self.durations * sampling_rate).astype(int)

        for i, row in enumerate(self.values.values):
            file_id = self.entities['event_file_id'].values[i]
            run_onset = self.collection.event_files[file_id].start
            ev_start = onsets[i] + int(math.ceil(run_onset * sampling_rate))
            ev_end = ev_start + durations[i]
            ts[ev_start:ev_end] = row

        ts = pd.DataFrame(ts)

        return DenseBIDSColumn(self.collection, self.name, ts)

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
                             amp: self.values.values})
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
            col = SparseBIDSColumn(self.collection, name, g, level_name=name,
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
        return self.collection.dense_index

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
        return [DenseBIDSColumn(self.collection, '%s/%s' % (self.name, name),
                                df[name].values)
                for i, name in enumerate(names)]


BIDSEventFile = namedtuple('BIDSEventFile', ('image_file', 'event_file',
                                             'start', 'duration', 'entities'))


class BIDSEventCollection(object):

    ''' A container for all design events extracted from a BIDS project.
    Args:
        base_dir (str): Path to the root of the BIDS project.
        default_duration (float): Default duration to assign to events in cases
            where a duration is missing.
        default_amplitude (float): Default amplitude to assign to events in
            cases where an amplitude is missing.
        condition_column (str): Optional name of the column that contains
            the names of conditions. Defaults to 'trial_type', per the BIDS
            specification. If None, only extra columns (beyond onset and
            duration) are inspected for events.
        amplitude_column (str): Optional name of the column providing the
            amplitudes that correspond to the 'condition' column. Ignored if
            condition_column is None.
        entities (list): Optional list of entities to encode and index by.
            If None (default), uses the standard BIDS-defined entities of
            ['run', 'session', 'subject', 'task'].
        extra_columns (list): Optional list of names specifying which extra
            columns in the event files to read. By default, reads all columns
            found.
        sampling_rate (int, float): Sampling rate to use internally when
            converting sparse columns to dense format.
    '''

    def __init__(self, base_dir, default_duration=0, default_amplitude=1,
                 condition_column='trial_type', amplitude_column=None,
                 entities=None, extra_columns=True, sampling_rate=10):

        self.default_duration = default_duration
        self.default_amplitude = default_amplitude
        self.condition_column = condition_column
        self.amplitude_column = amplitude_column
        self.base_dir = base_dir
        self.project = BIDSLayout(self.base_dir)
        if entities is None:
            entities = ['subject', 'session', 'task', 'run']
        self.entities = entities
        self.extra_columns = extra_columns
        self.sampling_rate = sampling_rate

    def read(self, file_directory=None, reset=True, **kwargs):
        ''' Read in and process event files.
        Args:
            file_directory (None, str): Optional path to folder containing
                event files to read from. If None (default), will use all event
                files found within the current BIDS project. Note that if
                `files` is passed, existing event files within the BIDS
                project are completely ignored.
            reset (bool): If True (default), clears all previously processed
                event files and columns; if False, adds new files
                incrementally.

        Notes:
            If a directory of event files is passed in using the `file_directory`
            argument, the following two constraints apply:
                * ALL relevant BIDS entities MUST be encoded in each file.
                  I.e., "sub-02_task-mixedgamblestask_run-01_events.tsv" is
                  valid, but "subject_events.tsv" would not be.
                * It is assumed that all and only the event files in the folder
                  are to be processed--i.e., there cannot be any other
                  subjects, runs, etc. whose events also need to be processed
                  but that are missing from the passed list.
        '''
        if reset:
            self.event_files = []
            self.columns = {}

        valid_files = []

        # Starting with either files or images, get all event files that have
        # a valid functional run, and store their duration if available.
        if file_directory is not None:
            new_project = BIDSLayout(file_directory)
            evf = new_project.get(return_type='file', extensions='.tsv',
                                   type='events', **kwargs)
            for f in evf:
                f_ents = new_project.files[f].entities
                f_ents = {k: v for k, v in f_ents.items() if k in self.entities}

                img_f = self.project.get(return_type='file', modality='func',
                                         extensions='.nii.gz', type='bold',
                                         **f_ents)

                if not img_f:
                    continue
                elif len(img_f) > 1:
                    warnings.warn("Event file matched multiple images,"
                                  "matching to first")

                valid_files.append((f, img_f[0], f_ents))

        else:

            images = self.project.get(return_type='file', modality='func',
                                      type='bold', extensions='.nii.gz',
                                      **kwargs)
            if not images:
                raise Exception("No functional runs found in BIDS project.")

            for img_f in images:
                f_ents = self.project.files[img_f].entities
                f_ents = {k: v for k, v in f_ents.items() if k in self.entities}

                evf = self.project.get(return_type='file', extensions='.tsv',
                                       type='events', **f_ents)

                if not evf:
                    continue

                valid_files.append((evf[0], img_f, f_ents))

        dfs = []
        start_time = 0

        for evf, img_f, f_ents in valid_files:

            _data = pd.read_table(evf, sep='\t')
            _data = _data.replace('n/a', np.nan)  # Replace BIDS' n/a
            _data = _data.apply(pd.to_numeric, errors='ignore')

            # Get duration of run: first try to get it directly from the image
            # header; if that fails, fall back on taking the offset of the last
            # event in the event file--but raise warning, because this is
            # suboptimal (e.g., there could be empty volumes at the end).
            try:
                img = nb.load(img_f)
                duration = img.shape[3] * img.header.get_zooms()[-1] / 1000
            except:
                duration = (_data['onset'] + _data['duration']).max()
                # warnings.warn("Unable to extract scan duration from image %s; "
                #               "setting duration to the offset of the last "
                #               "detected event instead (%d)--but note that this"
                #               " may produce unexpected results." %
                #               (img_f, duration))

            evf_index = len(self.event_files)
            f_ents['event_file_id'] = evf_index
            ef = BIDSEventFile(event_file=evf, image_file=img_f,
                               start=start_time, duration=duration,
                               entities=f_ents)
            self.event_files.append(ef)
            start_time += duration

            skip_cols = ['onset', 'duration']

            file_df = []

            # If condition column is provided, either extract amplitudes
            # from given amplitude column, or to default value
            if self.condition_column is not None:
                if self.condition_column not in _data.columns:
                    warnings.warn(
                        "Event file is missing the specified"
                        "condition column, {}. Setting to None".format(
                            self.condition_column))
                    self.condition_column = None
                else:
                    skip_cols.append(self.condition_column)

                    if self.amplitude_column is not None:
                        if self.amplitude_column not in _data.columns:
                            raise ValueError(
                                "Event file is missing the specified "
                                "amplitude column, {}".format(
                                    self.amplitude_column))
                        else:
                            amplitude = _data[self.amplitude_column]
                            skip_cols.append(self.amplitude_column)
                    else:
                        if 'amplitude' in _data.columns:
                            warnings.warn("Setting amplitude to values in "
                                          "column 'amplitude'")
                            amplitude = _data['amplitude']
                            skip_cols.append('amplitude')
                        else:
                            amplitude = _data[self.condition_column]

                    df = _data[['onset', 'duration']].copy()
                    df['condition'] = _data[self.condition_column]
                    df['amplitude'] = amplitude
                    df['factor'] = self.condition_column

                    file_df.append(df)

            extra = self.extra_columns
            if extra:
                if not isinstance(extra, (list, tuple)):
                    extra = list(set(_data.columns.tolist()) - set(skip_cols))

                for col in extra:
                    df = _data[['onset', 'duration']].copy()
                    df['condition'] = col
                    df['amplitude'] = _data[col].values
                    df['factor'] = col
                    file_df.append(df.dropna())

            _df = pd.concat(file_df, axis=0)
            for entity, value in f_ents.items():
                _df[entity] = value

            dfs.append(_df)

        _df = pd.concat(dfs, axis=0)

        for name, grp in _df.groupby(['factor', 'condition']):
            self._create_column(name, grp)

        # build the index
        self._build_dense_index()

    def write(self, path=None, file=None, suffix='_events', columns=None,
              sparse=True, sampling_rate=None, header=True, overwrite=False):
        ''' Write out all events in collection to TSV file(s).
        Args:
            path (str): The directory to write event files to
            file (str): Event file to write events to
            suffix (str): Suffix to append to filenames
            columns (list): Optional list of column names to retain; if None,
                all columns are written out.
            sparse (bool): If True, events will be written out in sparse format
                provided they are all internally represented as such. If False,
                a dense matrix (i.e., uniform sampling rate for all events)
                will be exported. Will be ignored if at least one column is
                dense.
            sampling_rate (float): If a dense matrix is written out, the sampling
                rate (in Hz) to use for downsampling. Defaults to the value
                currently set in the instance.
            header (bool): If True, includes column names in the header row. If
                False, omits the header row.
            overwrite (bool): If True, any existing .tsv file at the output
                location will be overwritten.
        '''

        if path is None and file is None:
            raise ValueError("Either the 'path' or the 'file' arguments must "
                             "be provided.")

        # Can only write sparse output if all columns are sparse
        force_dense = True if not sparse or not self._all_sparse() else False

        # Default to sampling rate of current instance
        # TODO: Store the TR internally when reading images, and then allow
        # user to downsample to TR resolution using the special value 'tr'.
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if force_dense and not self._all_dense():
            _cols = self.resample(sampling_rate, force_dense=force_dense,
                                    in_place=False)
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

        # By default drop columns for internal use
        _drop_cols = [c for c in data.columns if c in ['event_file_id', 'time']]
        # If output is a single file, just write out the entire DF, adding in
        # the entities.
        if file is not None:
            data.drop(_drop_cols, axis=1).\
                 to_csv(file, sep='\t', header=header, index=False)

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
                g.drop(_drop_cols, axis=1).\
                  to_csv(filename, sep='\t', header=header, index=False)

    def _all_sparse(self):
        return all([isinstance(c, SparseBIDSColumn) for c in self.columns.values()])

    def _all_dense(self):
        return all([isinstance(c, DenseBIDSColumn) for c in self.columns.values()])

    def _create_column(self, name, data):
        # If amplitude column contains categoricals, split on it and create
        # 1 dummy-coded column per level
        data = data.apply(pd.to_numeric, errors='ignore')
        factor, condition = name[0], name[1]
        if data['amplitude'].dtype.kind not in 'bifc':
            grps = data.groupby('amplitude')
            for i, (lev_name, lev_grp) in enumerate(grps):
                name = '%s/%s' % (factor, lev_name)
                lev_grp['amplitude'] = self.default_amplitude
                col = SparseBIDSColumn(self, name, lev_grp,
                                       factor_name=factor, factor_index=i,
                                       level_name=lev_name)
                self.columns[name] = col
        else:
            self.columns[condition] = SparseBIDSColumn(self, condition, data)

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
        sr = int(1000./self.sampling_rate)
        for evf in self.event_files:
            reps = int(math.ceil(evf.duration * self.sampling_rate))
            ent_vals = list(evf.entities.values())
            data = np.broadcast_to(ent_vals, (reps, len(ent_vals)))
            df = pd.DataFrame(data, columns=list(evf.entities.keys()))
            df['time'] = pd.date_range(0, periods=len(df), freq='%sms' % sr)
            index.append(df)
        self.dense_index = pd.concat(index, axis=0).reset_index(drop=True)

    def apply(self, func, cols, *args, **kwargs):
        ''' Applies an arbitrary callable or named function. Mostly useful for
        automating transformations via an external spec.
        Args:
            func (str, callable): ither a callable, or a string giving the
                name of an existing bound method to apply.
            args, kwargs: Optional positional and keyword arguments to pass
                on to the callable.
        '''
        if isinstance(func, string_types):
            if not hasattr(transform, func):
                raise ValueError("No transformation '%s' found!" % func)
            func = getattr(transform, func)

        func(self, cols, *args, **kwargs)

    def apply_from_json(self, spec):
        ''' Apply a series of transformations from a JSON spec.
        spec (str): Path to the JSON file containing transformations.
        '''
        if isinstance(spec, str) and os.path.exists(spec):
            spec = json.load(open(spec, 'rU'))
        for t in spec['transformations']:
            name = t.pop('name')
            cols = t.pop('input', None)
            self.apply(name, cols, **t)

    def resample(self, sampling_rate, force_dense=False, in_place=True,
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
        old_sr = self.sampling_rate
        n = len(self.dense_index)

        # Rebuild the dense index
        self.sampling_rate = sampling_rate
        self._build_dense_index()

        x = np.arange(n)
        num = n * sampling_rate / old_sr

        columns = {}

        for name, col in self.columns.items():
            if isinstance(col, SparseBIDSColumn):
                if force_dense:
                    columns[name] = col.to_dense()
            else:
                col = col.clone()
                f = interp1d(x, col.values.values.ravel(), kind=kind)
                x_new = np.linspace(0, n-1, num=num)
                col.values = pd.DataFrame(f(x_new))
                columns[name] = col

        if in_place:
            for k, v in columns.items():
                self.columns[k] = v
        else:
            return columns.values()
