import numpy as np

from ..hrf import compute_regressor


def test_duplicate_onsets():
    onset = [
        16,
        24,
        48,
        79,
        80,
        112,
        127,
        144,
        176,
        176,
        208,
        216,
        240,
        272,
        272,
        304,
        329,
    ]
    duration = [16, 1, 16, 1, 16, 16, 1, 16, 1, 16, 16, 1, 16, 1, 16, 16, 1]

    trial_types = {
        "Face": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "Hand": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        "Swallow": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    }

    model = "spm"
    resample_frames = np.arange(0, 340.0, 0.1)

    for name, trial_type in trial_types.items():
        vals = np.vstack([onset, duration, trial_type])
        regressor, _ = compute_regressor(
            vals, model, resample_frames, fir_delays=None, min_onset=0, oversampling=1
        )

        assert np.all((regressor > -0.2) & (regressor < 1.2))
