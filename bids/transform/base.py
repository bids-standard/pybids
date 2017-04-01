import pandas as pd
import numpy as np
from bids.grabbids import BIDSLayout
import nibabel as nb
import warnings
from collections import namedtuple
import math
from copy import deepcopy
from abc import abstractproperty, ABCMeta


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
            result.values = pd.Series(data, name=self.values.name)
        if kwargs:
            for k, v in kwargs.items():
                setattr(result, k, v)
        return result

    def __deepcopy__(self, memo):
        # When deep copying, we want to stay linked to the same collection.
        dc_method = self.__deepcopy__
        self.__deepcopy__ = None
        coll = self.collection
        self.collection = None
        clone = deepcopy(self, memo)
        clone.collection = coll
        self.collection = coll
        self.__deepcopy__ = dc_method
        clone.__deepcopy__ = dc_method
        return clone

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

    def to_dense(self, transformer):
        ''' Convert the current sparse column to a dense representation.
        Args:
            transformer (BIDSTransformer): A transformer object containing
                information controlling the densification process.
        '''

        if isinstance(self, DenseBIDSColumn):
            return self

        sampling_rate = transformer.sampling_rate
        duration = len(transformer.dense_index)
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

        return DenseBIDSColumn(self.collection, self.name, ts, transformer)

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.entities


class DenseBIDSColumn(BIDSColumn):
    ''' A dense representation of a single column. '''

    def __init__(self, collection, name, values, transformer):
        self.transformer = transformer
        super(DenseBIDSColumn, self).__init__(collection, name, values)

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.transformer.dense_index


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
        kwargs: Optional keywords to pass onto the read() call (just for
            convenience).
    '''

    def __init__(self, base_dir, default_duration=0, default_amplitude=1,
                 condition_column='trial_type', amplitude_column=None,
                 entities=None, extra_columns=True):

        self.default_duration = default_duration
        self.default_amplitude = default_amplitude
        self.condition_column = condition_column
        self.amplitude_column = amplitude_column
        self.base_dir = base_dir
        self.project = BIDSLayout(self.base_dir)
        if entities is None:
            entities = ['run', 'session', 'subject', 'task']
        self.entities = entities
        self.extra_columns = extra_columns

    def read(self, files=None, reset=True, **kwargs):
        ''' Read in and process event files.
        Args:
            files (None, list): Optional list of event files to read from. If
                None (default), will use all event files found within the
                current BIDS project. Note that if `files` is passed, existing
                event files within the BIDS project are completely ignored.
            reset (bool): If True (default), clears all previously processed
                event files and columns; if False, adds new files
                incrementally.

        Notes:
            If the names of event files are passed in using the `files`
            argument, the following two constraints apply:
                * ALL relevant BIDS entities MUST be encoded in each file.
                  I.e., "sub-02_task-mixedgamblestask_run-01_events.tsv" is
                  valid, but "subject_events.tsv" would not be.
                * It is assumed that all and only the files in the passed list
                  are to be processed--i.e., there cannot be any other
                  subjects, runs, etc. whose events also need to be processed
                  but that are missing from the passed list.
        '''
        if reset:
            self.event_files = []
            self.columns = {}

        valid_pairs = []

        # Starting with either files or images, get all event files that have
        # a valid functional run, and store their duration if available.
        if files is not None:

            for f in files:
                f_ents = self.project.files[f].entities
                f_ents = {k: v for k, v in f_ents.items() if k in self.entities}

                img_f = self.project.get(return_type='file', modality='func',
                                         extensions='.nii.gz', type='bold',
                                         **kwargs)

                if not img_f:
                    continue

                valid_pairs.append((f, img_f[0]))

        else:

            images = self.project.get(return_type='file', modality='func',
                                      type='bold', extensions='.nii.gz',
                                      **kwargs)
            if not images:
                raise Exception("No functional runs found in BIDS project.")

            for img_f in images:
                f_ents = self.project.files[img_f].entities
                f_ents = {k: v for k, v in f_ents.items() if k in self.entities}

                # HARDCODED FOR DEVELOPMENT
                # TODO: need to walk up the tree for each image to get .tsv
                # file; can't assume that entities will always be set in the
                # filename.
                evf = self.project.get(return_type='file', extensions='.tsv',
                                       type='events', **f_ents)

                # evf = self.project.get_event_file(img_f) # NOT IMPLEMENTED YET!!!
                if not evf:
                    continue

                valid_pairs.append((evf[0], img_f))

        dfs = []
        start_time = 0

        for evf, img_f in valid_pairs:

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
                warnings.warn("Unable to extract scan duration from image %s; "
                              "setting duration to the offset of the last "
                              "detected event instead (%d)--but note that this"
                              " may produce unexpected results." %
                              (img_f, duration))

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

                skip_cols.append(self.condition_column)

                if self.condition_column not in _data.columns:
                    raise ValueError(
                        "Event file is missing the specified"
                        "condition column, {}".format(self.condition_column))

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
                        warnings.warn("Setting amplitude to values in column "
                                      "'amplitude'")
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
        self.columns[col] = obj
