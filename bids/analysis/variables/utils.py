import math
import numpy as np
import pandas as pd


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
