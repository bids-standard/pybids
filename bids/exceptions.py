""" Exceptions.

Exceptions relating to problems with BIDS itself, should carry BIDS in their
name.  All exceptions should subclass from PyBIDSError
"""


class PyBIDSError(Exception):
    """ Base class.  Typically for mix-in."""


class ConfigError(ValueError, PyBIDSError):
    """ Problems with config file. """


class NoMatchError(ValueError, PyBIDSError):
    """ No match found where it is required. """


class BIDSEntityError(AttributeError, PyBIDSError):
    """ An unknown entity. """


class TargetError(ValueError, PyBIDSError):
    """ An unknown target. """


class BIDSValidationError(ValueError, PyBIDSError):
    """ An invalid BIDS dataset. """


class BIDSDerivativesValidationError(BIDSValidationError):
    """ An invalid BIDS derivative dataset. """


class BIDSConflictingValuesError(BIDSValidationError):
    """ A value conflict (e.g. in filename and side-car .json) """
