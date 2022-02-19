"""
The PyBIDS extension namespace package

``bids.ext`` is reserved as a namespace for extensions to install into.
To write such an extension, the following things are needed:

1) Create a new package with the following structure (assuming setuptools)::

    package/
      bids/
        ext/
          __init__.py
          EXTENSION/
            __init__.py
            ...
      setup.cfg
      setup.py

  The important things to note are that the ``bids/`` directory must be
  empty apart from ``ext/`` and ``bids/ext/`` must be empty except for
  your extension and an ``__init__.py``.

2) Place the following (and nothing else) in ``__init__.py``::

    __path__ = __import__('pkgutil').extend_path(__path__, __name__)

3) Include the following lines in ``setup.cfg``::

    [options]
    install_requires =
        pybids >= 0.15
    packages = find_namespace:

    [options.packages.find]
    include =
        bids.ext.EXTENSION
        bids.ext.EXTENSION.*

"""

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
