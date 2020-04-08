class BIDSMetadata(dict):
    """ Metadata dictionary that reports the associated file on lookup failures. """
    def __init__(self, source_file):
        self._source_file = source_file
        super().__init__()

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise KeyError(
                "Metadata term {!r} unavailable for file {}.".format(key, self._source_file))
