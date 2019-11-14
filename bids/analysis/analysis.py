import json
from io import open
from bids.layout import BIDSLayout
from bids.utils import matches_entities, convert_JSON
from bids.variables import BIDSVariableCollection, merge_collections
from . import transformations as transform
from collections import namedtuple, OrderedDict
import numpy as np
import pandas as pd
from itertools import chain


class Analysis(object):
    """Represents an entire BIDS-Model analysis.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout` or str
        A BIDSLayout instance or path to pass on
        to the BIDSLayout initializer.
    model : str or dict
        A BIDS model specification. Can either be a
        string giving the path of the JSON model spec, or an already-loaded
        dict containing the model info.
    """

    def __init__(self, layout, model):

        if not isinstance(layout, BIDSLayout):
            layout = BIDSLayout(layout)
        self.layout = layout

        self._load_model(model)

    def __iter__(self):
        for b in self.steps:
            yield b

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.steps[index]
        level = index.lower()
        name_matches = list(filter(lambda x: x.name == level, self.steps))
        if not name_matches:
            raise KeyError('There is no block with the name "%s".' % index)
        return name_matches[0]


    def _load_model(self, model):
        if isinstance(model, str):
            with open(model, 'r', encoding='utf-8') as fobj:
                model = json.load(fobj)

        # Convert JSON from CamelCase to snake_case keys
        self.model = convert_JSON(model)

        steps = self.model['steps']
        self.steps = []
        for i, step_args in enumerate(steps):
            step_args['level'] = step_args['level'].lower()
            step = Step(self.layout, index=i, **step_args)
            self.steps.append(step)

    def setup(self, steps=None, drop_na=False, **kwargs):
        """Set up the sequence of steps for analysis.

        Parameters
        ----------
        steps : list
            Optional list of steps to set up. Each element
            must be either an int giving the index of the step in the
            JSON config block list, or a str giving the (unique) name of
            the step, as specified in the JSON config. Steps that do not
            match either index or name will be skipped.
        drop_na : bool
            Boolean indicating whether or not to automatically
            drop events that have a n/a amplitude when reading in data
            from event files.
        """

        # In the beginning, there was nothing
        input_nodes = None

        # Use inputs from model, and update with kwargs
        selectors = self.model.get('input', {}).copy()
        selectors.update(kwargs)

        for i, b in enumerate(self.steps):

            # Skip any steps whose names or indexes don't match block list
            if steps is not None and i not in steps and b.name not in steps:
                continue

            b.setup(input_nodes, drop_na=drop_na, **selectors)
            input_nodes = b.output_nodes


class Step(object):
    """Represents a single analysis block from a BIDS-Model specification.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        The BIDSLayout containing all project files.
    level : str
        The BIDS keyword to use as the grouping variable; must be
        one of ['run', 'session', 'subject', or 'dataset'].
    index : int
        The numerical index of the current Step within the
        sequence of steps.
    name : str
        Optional name to assign to the block. Must be specified
        in order to enable name-based indexing in the parent Analysis.
    transformations : list
        List of BIDS-Model transformations to apply.
    model : dict
        The 'model' part of the BIDS-Model block specification.
    contrasts : list
        List of contrasts to apply to the parameter estimates
        generated when the model is fit.
    input_nodes : list
        Optional list of AnalysisNodes to use as input to
        this Step (typically, the output from the preceding Step).
    dummy_contrasts : dict
        Optional dictionary specifying which conditions to create
        indicator contrasts for. Dictionary must include a
        "type" key ('t' or 'FEMA'), and optionally a subset of "conditions".
        This parameter is over-written by the setting
        in setup() if the latter is passed.
    """

    def __init__(self, layout, level, index, name=None, transformations=None,
                 model=None, contrasts=None, input_nodes=None,
                 dummy_contrasts=False):
        self.layout = layout
        self.level = level.lower()
        self.index = index
        self.name = name
        self.transformations = transformations or []
        self.model = model or None
        self.contrasts = contrasts or []
        self.input_nodes = input_nodes or []
        self.dummy_contrasts = dummy_contrasts
        self.output_nodes = []

    def _filter_objects(self, objects, kwargs):
        # Keeps only objects that match target entities, and also removes those
        # keys from the kwargs dict.
        valid_ents = {'task', 'subject', 'session', 'run'}
        entities = {k: kwargs.pop(k) for k in dict(kwargs) if k in valid_ents}
        objects = [o for o in objects if o.matches_entities(entities)]
        return (objects, kwargs)

    def _group_objects(self, objects):
        # Group list of objects into bins defined by all entities at current
        # Step level or higher.
        if self.level == 'dataset':
            return [objects]
        groups = OrderedDict()
        valid_ents = ['subject', 'session', 'task', 'run']
        valid_ents = valid_ents[:(valid_ents.index(self.level) + 1)]
        for o in objects:
            key = {k: v for k, v in o.entities.items() if k in valid_ents}
            key = tuple(sorted(key.items(), key=str))
            if key not in groups:
                groups[key] = []
            groups[key].append(o)
        return list(groups.values())

    def _concatenate_input_nodes(self, nodes):
        data, entities = [], []
        for n in nodes:
            contrasts = [c.name for c in n.contrasts]
            row = pd.Series(np.ones(len(contrasts)), index=contrasts)
            data.append(row)
            entities.append(pd.Series(n.entities))
        data = pd.concat(data, axis=1, sort=True).T
        entities = pd.concat(entities, axis=1, sort=True).T
        return BIDSVariableCollection.from_df(data, entities, self.level)

    def setup(self, input_nodes=None, drop_na=False, **kwargs):
        """Set up the Step and construct the design matrix.

        Parameters
        ----------
        input_nodes : list
            Optional list of Node objects produced by
            the preceding Step in the analysis. If None, uses any inputs
            passed in at Step initialization.
        drop_na : bool
            Boolean indicating whether or not to automatically
            drop events that have a n/a amplitude when reading in data
            from event files.
        kwargs : dict
            Optional keyword arguments to pass onto load_variables.
        """
        self.output_nodes = []
        input_nodes = input_nodes or self.input_nodes or []

        # TODO: remove the scan_length argument entirely once we switch tests
        # to use the synthetic dataset with image headers.
        if self.level != 'run':
            kwargs = kwargs.copy()
            kwargs.pop('scan_length', None)

        collections = self.layout.get_collections(self.level, drop_na=drop_na,
                                                  **kwargs)
        objects = collections + input_nodes

        objects, kwargs = self._filter_objects(objects, kwargs)
        groups = self._group_objects(objects)

        # Set up and validate variable lists
        model = self.model or {}
        X = model.get('x', [])

        for grp in groups:
            # Split into separate lists of Collections and Nodes
            input_nodes = [o for o in grp if isinstance(o, AnalysisNode)]
            colls = list(set(grp) - set(input_nodes))

            if input_nodes:
                node_coll = self._concatenate_input_nodes(input_nodes)
                colls.append(node_coll)

            coll = merge_collections(colls) if len(colls) > 1 else colls[0]

            coll = apply_transformations(coll, self.transformations)
            if X:
                transform.Select(coll, X)

            node = AnalysisNode(self.level, coll, self.contrasts, input_nodes,
                                self.dummy_contrasts)

            self.output_nodes.append(node)

    def get_design_matrix(self, names=None, format='long', mode='both',
                          force=False, sampling_rate='TR', **kwargs):
        """Get design matrix and associated information.

        Parameters
        ----------
        names : list
            Optional list of names of variables to include in the
            returned design matrix. If None, all variables are included.
        format : str
            Whether to return the design matrix in 'long' or
            'wide' format. Note that dense design matrices are always
            returned in 'wide' format.
        mode : str
            Specifies whether to return variables in a sparse
            representation ('sparse'), dense representation ('dense'), or
            both ('both').
        force : bool
            Indicates how to handle columns not of the type
            indicated by the mode argument. When False, variables of the
            non-selected type will be silently ignored. When True,
            variables will be forced to the desired representation. For
            example, if mode='dense' and force=True, sparse variables will
            be converted to dense variables and included in the returned
            design matrix in the .dense attribute. The force argument is
            ignored entirely if mode='both'.
        sampling_rate : {'TR', 'highest'} or float
            Sampling rate at which to
            generate the dense design matrix. When 'TR', the repetition
            time is used, if available, to select the sampling rate (1/TR).
            When 'highest', all variables are resampled to the highest
            sampling rate of any variable. The sampling rate may also be
            specified explicitly in Hz. Has no effect on sparse design
            matrices.
        kwargs : dict
            Optional keyword arguments. Includes (1) selectors used
            to constrain which of the available nodes get returned
            (e.g., passing subject=['01', '02'] will return design
            information only for subjects '01' and '02'), and (2) arguments
            passed on to each Variable's to_df() call (e.g.,
            sampling_rate, entities, timing, etc.).

        Returns
        -------
        list of DesignMatrixInfo namedtuples
            one tuple per unit of the current
            analysis level (e.g., if level='run', each element in the list
            represents the design matrix for a single run).
        """
        nodes, kwargs = self._filter_objects(self.output_nodes, kwargs)
        return [n.get_design_matrix(names, format, mode=mode, force=force,
                                    sampling_rate=sampling_rate, **kwargs)
                for n in nodes]

    def get_contrasts(self, names=None, variables=None, **kwargs):
        """Return contrast information for the current block.

        Parameters
        ----------
        names : list
            Optional list of names of contrasts to return. If
            None (default), all contrasts are returned.
        variables : bool
            Optional list of strings giving the names of
            design matrix columns to use when generating the matrix of
            weights.
        kwargs : dict
            Optional keyword arguments used to constrain which of the
            available nodes get returned (e.g., passing subject=['01',
            '02'] will return contrast  information only for subjects '01'
            and '02').

        Returns
        -------
        list
            A list with one element per unit of the current analysis level
            (e.g., if level='run', each element in the list representing the
            contrast information for a single run). Each element is a list of
            ContrastInfo namedtuples (one per contrast).

        """
        nodes, kwargs = self._filter_objects(self.output_nodes, kwargs)
        return [n.get_contrasts(names, variables) for n in nodes]


DesignMatrixInfo = namedtuple('DesignMatrixInfo',
                              ('sparse', 'dense', 'entities'))


ContrastInfo = namedtuple('ContrastInfo', ('name', 'weights', 'type',
                                           'entities'))


class AnalysisNode(object):
    """A single analysis node generated within a Step.

    Parameters
    ----------
    level : str
        The level of the Node. Most be one of 'run', 'session',
        'subject', or 'dataset'.
    collection : :obj:`bids.variables.kollekshuns.BIDSVariableCollection`
        The BIDSVariableCollection containing variables at this Node.
    contrasts : list
        A list of contrasts defined in the originating Step.
    dummy_contrasts : list
        Optional dictionary specifying which conditions to create
        indicator contrasts for. Dictionary must include a
        "type" key ('t' or 'FEMA'), and optionally a subset of "conditions".
        This parameter is over-written by the setting
        in setup() if the latter is passed.
    """

    def __init__(self, level, collection, contrasts, input_nodes=None,
                 dummy_contrasts=None):
        self.level = level.lower()
        self.collection = collection
        self._block_contrasts = contrasts
        self.input_nodes = input_nodes
        self.dummy_contrasts = dummy_contrasts
        self._contrasts = None

    @property
    def entities(self):
        return self.collection.entities

    @property
    def contrasts(self):
        if self._contrasts is None:
            self.get_contrasts()
        return self._contrasts

    def get_design_matrix(self, names=None, format='long', mode='both',
                          force=False, sampling_rate='TR', **kwargs):
        """Get design matrix and associated information.

        Parameters
        ----------
        names : list
            Optional list of names of variables to include in the
            returned design matrix. If None, all variables are included.
        format : str
            Whether to return the design matrix in 'long' or
            'wide' format. Note that dense design matrices are always
            returned in 'wide' format.
        mode : str
            Specifies whether to return variables in a sparse
            representation ('sparse'), dense representation ('dense'), or
            both ('both').
        force : bool
            Indicates how to handle columns not of the type
            indicated by the mode argument. When False, variables of the
            non-selected type will be silently ignored. When True,
            variables will be forced to the desired representation. For
            example, if mode='dense' and force=True, sparse variables will
            be converted to dense variables and included in the returned
            design matrix in the .dense attribute. The force argument is
            ignored entirely if mode='both'.
        sampling_rate : {'TR', 'highest'} or float
            Sampling rate at which to
            generate the dense design matrix. When 'TR', the repetition
            time is used, if available, to select the sampling rate (1/TR).
            When 'highest', all variables are resampled to the highest
            sampling rate of any variable. The sampling rate may also be
            specified explicitly in Hz. Has no effect on sparse design
            matrices.
        kwargs : dict
            Optional keyword arguments to pass onto each Variable's
            to_df() call (e.g., sampling_rate, entities, timing, etc.).

        Returns
        -------
        A DesignMatrixInfo namedtuple.
        """
        sparse_df, dense_df = None, None
        coll = self.collection

        if self.level != 'run' and mode != 'sparse':
            mode = 'sparse'

        include_sparse = include_dense = (force and mode != 'both')

        if mode in ['sparse', 'both']:
            kwargs['sparse'] = True
            sparse_df = coll.to_df(names, format, include_dense=include_dense,
                                   **kwargs)

        if mode in ['dense', 'both']:
            # The current implementation of pivoting to wide in
            # BIDSVariableCollection.to_df() breaks if we don't have the
            # temporal columns to index on, so we force their inclusion first
            # and then drop them afterwards.
            kwargs['timing'] = True
            kwargs['sparse'] = False

            if sampling_rate == 'TR':
                trs = {var.run_info[0].tr for var in self.collection.variables.values()}
                if not trs:
                    raise ValueError("Repetition time unavailable; specify sampling_rate "
                                     "explicitly")
                elif len(trs) > 1:
                    raise ValueError("Non-unique Repetition times found ({!r}); specify "
                                     "sampling_rate explicitly")
                sampling_rate = 1. / trs.pop()
            elif sampling_rate == 'highest':
                sampling_rate = None
            dense_df = coll.to_df(names, format='wide',
                                  include_sparse=include_sparse,
                                  sampling_rate=sampling_rate, **kwargs)
            if dense_df is not None:
                dense_df = dense_df.drop(['onset', 'duration'], axis=1)

        return DesignMatrixInfo(sparse_df, dense_df, self.entities)

    def get_contrasts(self, names=None, variables=None):
        """Return contrast information for the current block.

        Parameters
        ----------
        names : list
            Optional list of names of contrasts to return. If
            None (default), all contrasts are returned.
        variables : bool
            Optional list of strings giving the names of
            design matrix columns to use when generating the matrix of
            weights.

        Returns
        -------
        list
            A list of ContrastInfo namedtuples, one per contrast.

        Notes
        -----
        The 'variables' argument take precedence over the natural process
        of column selection. I.e.,
            if a variable shows up in a contrast, but isn't named in
            variables, it will *not* be included in the returned
        """

        # Verify that there are no invalid columns in the condition_lists
        all_conds = [c['condition_list'] for c in self._block_contrasts]
        all_conds = set(chain(*all_conds))
        bad_conds = all_conds - set(self.collection.variables.keys())
        if bad_conds:
            raise ValueError("Invalid condition names passed in one or more "
                             " contrast condition lists: %s." % bad_conds)

        # Construct a list of all contrasts, including dummy contrasts
        contrasts = list(self._block_contrasts)

        # Check that all contrasts have unique name
        contrast_names = [c['name'] for c in contrasts]
        if len(set(contrast_names)) < len(contrast_names):
            raise ValueError("One or more contrasts have the same name")
        contrast_names = list(set(contrast_names))

        if self.dummy_contrasts:
            if 'conditions' in self.dummy_contrasts:
                conditions = [c for c in self.dummy_contrasts['conditions']
                              if c in self.collection.variables.keys()]
            else:
                conditions = self.collection.variables.keys()

            for col_name in conditions:
                if col_name not in contrast_names:
                    contrasts.append({
                        'name': col_name,
                        'condition_list': [col_name],
                        'weights': [1],
                        'type': self.dummy_contrasts['type']
                    })

        # Filter on desired contrast names if passed
        if names is not None:
            contrasts = [c for c in contrasts if c['name'] in names]

        def setup_contrast(c):
            weights = np.atleast_2d(c['weights'])
            weights = pd.DataFrame(weights, columns=c['condition_list'])
            # If variables were explicitly passed, use them as the columns
            if variables is not None:
                var_df = pd.DataFrame(columns=variables)
                weights = pd.concat([weights, var_df],
                                    sort=True)[variables].fillna(0)

            test_type = c.get('type', ('t' if len(weights) == 1 else 'F'))

            return ContrastInfo(c['name'], weights, test_type, self.entities)

        self._contrasts = [setup_contrast(c) for c in contrasts]

        return self._contrasts

    def matches_entities(self, entities, strict=False):
        """Determine whether current AnalysisNode matches passed entities.

        Parameters
        ----------
        entities : dict
            Dictionary of entities to match. Keys are entity
            names; values are single values or lists.
        strict : bool
            If True, _all_ entities in the current Node must
            match in order to return True.
        """
        return matches_entities(self, entities, strict)


def apply_transformations(collection, transformations, select=None):
    """Apply all transformations to the variables in the collection.

    Parameters
    ----------
    transformations : list
        List of transformations to apply.
    select : list
        Optional list of names of variables to retain after all
        transformations are applied.
    """
    for t in transformations:
        kwargs = dict(t)
        func = kwargs.pop('name')
        cols = kwargs.pop('input', None)

        if isinstance(func, str):
            if func in ('and', 'or'):
                func += '_'
            if not hasattr(transform, func):
                raise ValueError("No transformation '%s' found!" % func)
            func = getattr(transform, func)
            func(collection, cols, **kwargs)

    if select is not None:
        transform.Select(collection, select)

    return collection
