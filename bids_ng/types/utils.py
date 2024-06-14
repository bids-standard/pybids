import typing as ty


class PaddedInt(int):
    """Integer type that preserves zero-padding

    Acts like an int in almost all ways except that string formatting
    will keep the original zero-padding. Numeric format specifiers will
    work with the integer value.

    >>> PaddedInt(1)
    1
    >>> p2 = PaddedInt("02")
    >>> p2
    02
    >>> str(p2)
    '02'
    >>> p2 == 2
    True
    >>> p2 in range(3)
    True
    >>> f"{p2}"
    '02'
    >>> f"{p2:s}"
    '02'
    >>> f"{p2!s}"
    '02'
    >>> f"{p2!r}"
    '02'
    >>> f"{p2:d}"
    '2'
    >>> f"{p2:03d}"
    '002'
    >>> f"{p2:f}"
    '2.000000'
    >>> {2: "val"}.get(p2)
    'val'
    >>> {p2: "val"}.get(2)
    'val'

    Note that arithmetic will break the padding.

    >>> str(p2 + 1)
    '3'
    """

    def __init__(self, val: ty.Union[str, int]) -> None:
        self.sval = str(val)
        if not self.sval.isdigit():
            raise TypeError(
                f"{self.__class__.__name__}() argument must be a string of digits "
                f"or int, not {val.__class__.__name__!r}"
            )

    def __eq__(self, val: object) -> bool:
        return val == self.sval or super().__eq__(val)

    def __str__(self) -> str:
        return self.sval

    def __repr__(self) -> str:
        return self.sval

    def __format__(self, format_spec: str) -> str:
        """Format a padded integer

        If a format spec can be used on a string, apply it to the zero-padded string.
        Otherwise format as an integer.
        """
        try:
            return format(self.sval, format_spec)
        except ValueError:
            return super().__format__(format_spec)

    def __hash__(self) -> int:
        return super().__hash__()
