import pandas as pd

def merge_rowise(left, right, on=None, how='outer', conflict_suffix='_conflict'):
        """ Merges two dataframes and resolves row-wise conflicts.
        If more than value for a column is found, picks value from left df.

        Args:
            left (DataFrame): Left DataFrame
            right (DataFrame): Right DataFrame
            on (list): Columns to merge on
            how (string): Merge method.
        Returns:
            Merged dataframe
        """
        merged = pd.merge(left, right, on=['onset', 'duration'], how='outer',
                          suffixes=('', '_conflict'))

        clash_keys = [k for k in merged.columns if k.endswith(conflict_suffix)]

        for key in clash_keys:
            keep_key = key[:-len(conflict_suffix)]
            merged[keep_key] = merged[keep_key].combine_first(merged[key])

        merged = merged.drop(columns=clash_keys)

        return merged
