import pandas as pd
import numpy as np
from bids.grabbids import BIDSLayout
import nibabel as nb
import warnings
from collections import namedtuple
import math
from .utils import listify
from copy import deepcopy


class BIDSColumn(object):

    def __init__(self, collection, name, values):
        self.collection = collection
        self.name = name
        self.values = values

    def clone(self, data=None, **kwargs):
        ''' Clone (deep copy) the current column, optionally replacing its
        data and/or any other attributes. '''
        result = deepcopy(self)
        if data is not None:
            result.values = pd.DataFrame(data, columns=self.values.columns)
        if kwargs:
            for k, v in kwargs.items():
                setattr(result, k, v)
        return result

    def __deepcopy__(self, memo):
        # We want to copy all attributes except the linked collection
        dc_method = self.__deepcopy__
        self.__deepcopy__ = None
        coll = self.collection
        self.collection = None
        clone = deepcopy(self, memo)
        clone.collection = coll
        self.__deepcopy__ = dc_method
        return clone

    def _get_grouper(self, df, groupby):

        # First, retrieve the actual pandas columns for all groupby variables.
        # These can come from either the index of BIDS entities (stored in df),
        # or from the current BIDSEventCollection (if the user is trying to
        # group on another named column).
        groupby = listify(groupby)

        def get_column(col):
            if col in df.columns:
                return df.loc[:, col]
            elif col in self.collection.columns:
                vals = self.collection.columns[col].values
                if vals.shape[1] > 1:
                    raise ValueError("Cannot group on categorical variables "
                                     "(error occured for column '%s')." % col)
                    return vals
            raise ValueError("Desired column '%s' in groupby argument could "
                             "not be found.")
        groupby = [get_column(c) for c in groupby]

        # # Next, we need to make sure that the groupby columns all align with
        # # one another.
        # if len(groupby) > 1:
        #     # If any of the columns are dense, all of them must be dense
        #     if any(['onset' not in g.columns for g in groupby]):

        df = pd.concat(groupby, axis=1)

        return pd.core.groupby._get_grouper(df, df.columns)[0]


class SparseBIDSColumn(BIDSColumn):

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

        return DenseBIDSColumn(self.collection, self.name, ts)

    def get_grouper(self, groupby='event_file_id'):
        return pd.core.groupby._get_grouper(self.entities, groupby)[0]

    def apply(self, func, groupby='event_file_id', *args, **kwargs):
        grouper = self.get_grouper(groupby)
        return self.values.groupby(grouper).apply(func, *args, **kwargs)


class DenseBIDSColumn(BIDSColumn):

    def get_grouper(self, groupby='event_file_id'):
        return pd.core.groupby._get_grouper(self.collection.dense_index,
                                            groupby)[0]

    def apply(self, func, groupby='event_file_id', *args, **kwargs):
        grouper = self.get_grouper(groupby)
        return self.values.groupby(grouper).apply(func, *args, **kwargs)


BIDSEventFile = namedtuple('BIDSEventFile', ('image_file', 'event_file',
                                             'start', 'duration', 'entities'))


class BIDSEventCollection(object):

    def __init__(self, base_dir, default_duration=0, default_amplitude=1,
                 amplitude_column=None, condition_column='trial_type',
                 entities=None, extra_columns=True, **kwargs):

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
        self.read(**kwargs)

    def read(self, reset=True, **kwargs):

        if reset:
            self.event_files = []
            self.columns = {}

        images = self.project.get(return_type='file', modality='func',
                                  extensions='.nii.gz', **kwargs)
        if not images:
            raise Exception("No functional runs found in BIDS project.")

        dfs = []
        start_time = 0

        for img_f in images:

            f_ents = self.project.files[img_f].entities
            f_ents = {k: v for k, v in f_ents.items() if k in self.entities}

            # HARDCODED FOR DEVELOPMENT
            evf = self.project.get(
                return_type='file', extensions='.tsv', **f_ents)
            # evf = self.project.get_event_file(img_f) # NOT IMPLEMENTED YET!!!
            if not evf:
                continue
            _data = pd.read_table(evf[0], sep='\t')
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
                col = SparseBIDSColumn(self, name, data,
                                       factor_name=factor, factor_index=i,
                                       level_name=lev_name)
                self.columns[name] = col
        else:
            self.columns[condition] = SparseBIDSColumn(self, condition, data)

    def __getitem__(self, col):
        return self.columns[col]

    def __setitem__(self, col, obj):
        self.columns[col] = obj
