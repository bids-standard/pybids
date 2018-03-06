def _merge_event_files(events):
    """ Merge TSV event files sequentially, giving preference to values from
    event files higher in the hierarchy

    Args:
        events (list): List of str paths to event files
    Returns:
        Merged dataframe
    """
    def _merge_rowise(left, right, on=None, how='outer'):
            """ Merges two dfs and resolves row-wise conflicts.
            If more than value for a column is found, picks value from left df.
            Args:
                left (DataFrame): Left DataFrame
                right (DataFrame): Right DataFrame
                on (list): Columns to merge on
                how (string): Merge method (passed to pandas)
            """
            conf_suff = '_#remove#'
            merged = pd.merge(left, right, on=['onset', 'duration'],
                              how=how, suffixes=('', conf_suff))

            clash_keys = [k for k in merged.columns if k.endswith(conf_suff)]

            for key in clash_keys:
                keep_key = key[:-len(conf_suff)]
                merged[keep_key] = merged[keep_key].combine_first(merged[key])

            merged = merged.drop(clash_keys, axis=1)

            return merged

    try:
        import pandas as pd
    except ImportError:
        raise ValueError("The library pandas as a dependency to return events"
                         "as a merged DataFrame")
    merged = None
    for e in events:
        e = pd.read_csv(e, delimiter='\t')
        if merged is None:
            merged = e
        else:
            merged = _merge_rowise(merged, e, on=['onset', 'duration'])
    events = merged

    return events
