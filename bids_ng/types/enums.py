from enum import Enum


class Query(Enum):
    """Special arguments for dataset querying

    * `Query.NONE` - The field MUST NOT be present
    * `Query.REQUIRED` - The field MUST be present, but may take any value
    * `Query.OPTIONAL` - The field MAY be present, and may take any value

    `Query.ANY` is a synonym for `Query.REQUIRED`. Its use is discouraged
    and may be removed in the future.
    """

    NONE = 1
    REQUIRED = ANY = 2
    OPTIONAL = 3
