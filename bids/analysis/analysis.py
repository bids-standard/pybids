import json
from bids.grabbids import BIDSLayout
from bids.utils import matches_entities
from bids.variables import BIDSVariableCollection, merge_collections
from . import transformations as transform
from collections import namedtuple, OrderedDict
from six import string_types
import numpy as np
import pandas as pd
from itertools import chain


class Analysis(object):
    ''' Represents an entire BIDS-Model analysis.

    Args:
        layout (BIDSLayout, str): A BIDSLayout instance or path to pass on
            to the BIDSLayout initializer.
        model (str or dict): a BIDS model specification. Can either be a
            string giving the path of the JSON model spec, or an already-loaded
            dict containing the model info.
        collections (dict): An optional dictionary containing all variable
            collections to use. Keys are level names ('run', 'session', etc.);
            values are lists of BIDSVariableCollections. If None, collections
            will be read from the current layout.
        sampling_rate (int): Optional sampling rate (in Hz) to use when
            resampling variables internally. If None, the package-wide default
            will be used.
        scan_length (float): Duration of scanning runs. Only necessary in
            cases where the nifti image files or image headers are not locally
            available.
    '''

    def __init__(self, layout, model, collections=None, sampling_rate=None,
                 scan_length=None):

        self.sampling_rate = sampling_rate
        self.collections = collections
        self.scan_length = scan_length

        if not isinstance(layout, BIDSLayout):
            layout = BIDSLayout(layout)
        self.layout = layout

        if isinstance(model, str):
            model = json.load(open(model))
        self.model = model

        self._load_blocks(model['blocks'])

    def __iter__(self):
        for b in self.blocks:
            yield b

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.blocks[index]
        name_matches = list(filter(lambda x: x.name == index, self.blocks))
        if not name_matches:
            raise KeyError('There is no block with the name "%s".' % index)
        return name_matches[0]

    def _load_blocks(self, blocks):
        self.blocks = []
        for i, block_args in enumerate(blocks):
            block = Block(self.layout, index=i, **block_args)
            self.blocks.append(block)

    def setup(self, blocks=None, agg_func='mean', identity_contrasts=True,
              **kwargs):
        ''' Set up the sequence of blocks for analysis.

        Args:
            blocks (list): Optional list of blocks to set up. Each element
                must be either an int giving the index of the block in the
                JSON config block list, or a str giving the (unique) name of
                the block, as specified in the JSON config. Blocks that do not
                match either index or name will be skipped.
            agg_func (str or Callable): The aggregation function to use when
                combining rows from the previous level of analysis. E.g.,
                when analyzing a 'subject'-level block, inputs coming from the
                'session' level are typically averaged to produce individual
                subject-level estimates. Must be either a string giving the
                name of a function recognized by apply() in pandas, or a
                Callable that takes a DataFrame as input and returns a Series
                or a DataFrame. NOTE: CURRENTLY UNIMPLEMENTED.
            identity_contrasts (bool): If True, a contrast is automatically
                created for each column in the design matrix.
        '''

        # In the beginning, there was nothing
        input_nodes = None

        for i, b in enumerate(self.blocks):

            # Skip any blocks whose names or indexes don't match block list
            if blocks is not None and i not in blocks and b.name not in blocks:
                continue

            b.setup(input_nodes, identity_contrasts, **kwargs)
            input_nodes = b.output_nodes


class Block(object):

    ''' Represents a single analysis block from a BIDS-Model specification.

    Args:
        layout (BIDSLayout): The BIDSLayout containing all project files.
        level (str): The BIDS keyword to use as the grouping variable; must be
            one of ['run', 'session', 'subject', or 'dataset'].
        index (int): The numerical index of the current Block within the
            sequence of blocks.
        name (str): Optional name to assign to the block. Must be specified
            in order to enable name-based indexing in the parent Analysis.
        transformations (list): List of BIDS-Model transformations to apply.
        model (dict): The 'model' part of the BIDS-Model block specification.
        contrasts (list): List of contrasts to apply to the parameter estimates
            generated when the model is fit.
        input_nodes (list): Optional list of AnalysisNodes to use as input to
            this Block (typically, the output from the preceding Block).
    '''

    def __init__(self, layout, level, index, name=None, transformations=None,
                 model=None, contrasts=None, input_nodes=None):

        self.layout = layout
        self.level = level
        self.index = index
        self.name = name
        self.transformations = transformations or []
        self.model = model or None
        self.contrasts = contrasts or []
        self.input_nodes = input_nodes or []
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
        # Block level or higher.
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
            contrasts = n.contrasts.data.T
            row = pd.Series(np.ones(len(contrasts)), index=contrasts.index)
            data.append(row)
            entities.append(pd.Series(n.entities))
        data = pd.concat(data, axis=1).T
        entities = pd.concat(entities, axis=1).T
        return BIDSVariableCollection.from_df(data, entities, self.level)

    def setup(self, input_nodes=None, identity_contrasts=True, **kwargs):
        ''' Set up the Block and construct the design matrix.

        Args:
            input_nodes (list): Optional list of Node objects produced by
                the preceding Block in the analysis. If None, uses any inputs
                passed in at Block initialization.
            identity_contrasts (bool): If True, a contrast is automatically
                created for each column in the design matrix.
            kwargs: Optional keyword arguments to pass onto load_variables.
        '''

        input_nodes = input_nodes or self.input_nodes or []

        # TODO: remove the scan_length argument entirely once we switch tests
        # to use the synthetic dataset with image headers.
        if self.level != 'run':
            kwargs = kwargs.copy()
            kwargs.pop('scan_length', None)

        collections = self.layout.get_collections(self.level, **kwargs)
        objects = collections + input_nodes

        objects, kwargs = self._filter_objects(objects, kwargs)
        groups = self._group_objects(objects)

        for grp in groups:
            # Split into separate lists of Collections and Nodes
            input_nodes = [o for o in grp if isinstance(o, AnalysisNode)]
            colls = list(set(grp) - set(input_nodes))

            if input_nodes:
                node_coll = self._concatenate_input_nodes(input_nodes)
                colls.append(node_coll)

            coll = merge_collections(colls) if len(colls) > 1 else colls[0]
            coll = apply_transformations(coll, self.transformations)
            node = AnalysisNode(self.level, coll, self.contrasts, input_nodes,
                                identity_contrasts)

            self.output_nodes.append(node)

    def get_design_matrix(self, names=None, format='long', mode='both',
                          force=False, **kwargs):
        ''' Get design matrix and associated information.

        Args:
            names (list): Optional list of names of variables to include in the
                returned design matrix. If None, all variables are included.
            format (str): Whether to return the design matrix in 'long' or
                'wide' format. Note that dense design matrices are always
                returned in 'wide' format.
            mode (str): Specifies whether to return variables in a sparse
                representation ('sparse'), dense representation ('dense'), or
                both ('both').
            force (bool): Indicates how to handle columns not of the type
                indicated by the mode argument. When False, variables of the
                non-selected type will be silently ignored. When True,
                variables will be forced to the desired representation. For
                example, if mode='dense' and force=True, sparse variables will
                be converted to dense variables and included in the returned
                design matrix in the .dense attribute. The force argument is
                ignored entirely if mode='both'.
            kwargs: Optional keyword arguments. Includes (1) selectors used
                to constrain which of the available nodes get returned
                (e.g., passing subject=['01', '02'] will return design
                information only for subjects '01' and '02'), and (2) arguments
                passed on to each Variable's to_df() call (e.g.,
                sampling_rate, entities, timing, etc.).

        Returns:
            A list of DesignMatrixInfo namedtuples--one per unit of the current
            analysis level (e.g., if level='run', each element in the list
            represents the design matrix for a single run).
        '''
        nodes, kwargs = self._filter_objects(self.output_nodes, kwargs)
        return [n.get_design_matrix(names, format, mode=mode, force=force,
                                    **kwargs) for n in nodes]

    def get_contrasts(self, names=None, **kwargs):
        ''' Return contrast information for the current block.

        Args:
            names (list): Optional list of names of contrasts to return. If
                None (default), all contrasts are returned.
            kwargs: Optional keyword arguments used to constrain which of the
                available nodes get returned (e.g., passing subject=['01',
                '02'] will return contrast  information only for subjects '01'
                and '02').

        Returns:
            A list of ContrastMatrixInfo namedtuples--one per unit of the
            current analysis level (e.g., if level='run', each element in the
            list represents the design matrix for a single run).
        '''
        nodes, kwargs = self._filter_objects(self.output_nodes, kwargs)
        return [n.get_contrasts(names) for n in nodes]


DesignMatrixInfo = namedtuple('DesignMatrixInfo',
                              ('sparse', 'dense', 'entities'))


ContrastMatrixInfo = namedtuple('ContrastMatrixInfo', ('data', 'index',
                                                       'entities'))


class AnalysisNode(object):
    ''' A single analysis node generated within a Block.

    Args:
        level (str): The level of the Node. Most be one of 'run', 'session',
            'subject', or 'dataset'.
        collection (BIDSVariableCollection): The BIDSVariableCollection
            containing variables at this Node.
        contrasts (list): A list of contrasts defined in the originating Block.
        identity_contrasts (bool): If True, a contrast is automatically
            created for each column in the design matrix.
    '''

    def __init__(self, level, collection, contrasts, input_nodes=None,
                 identity_contrasts=True):
        self.level = level
        self.collection = collection
        self._block_contrasts = contrasts
        self.input_nodes = input_nodes
        self.identity_contrasts = identity_contrasts
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
                          force=False, **kwargs):
        ''' Get design matrix and associated information.

        Args:
            names (list): Optional list of names of variables to include in the
                returned design matrix. If None, all variables are included.
            format (str): Whether to return the design matrix in 'long' or
                'wide' format. Note that dense design matrices are always
                returned in 'wide' format.
            mode (str): Specifies whether to return variables in a sparse
                representation ('sparse'), dense representation ('dense'), or
                both ('both').
            force (bool): Indicates how to handle columns not of the type
                indicated by the mode argument. When False, variables of the
                non-selected type will be silently ignored. When True,
                variables will be forced to the desired representation. For
                example, if mode='dense' and force=True, sparse variables will
                be converted to dense variables and included in the returned
                design matrix in the .dense attribute. The force argument is
                ignored entirely if mode='both'.
            kwargs: Optional keyword arguments to pass onto each Variable's
                to_df() call (e.g., sampling_rate, entities, timing, etc.).

        Returns:
            A DesignMatrixInfo namedtuple.
        '''
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
            dense_df = coll.to_df(names, format='wide',
                                  include_sparse=include_sparse, **kwargs)
            if dense_df is not None:
                dense_df = dense_df.drop(['onset', 'duration'], axis=1)

        return DesignMatrixInfo(sparse_df, dense_df, self.entities)

    def get_contrasts(self, names=None, entities=False):
        ''' Return contrast information for the current block.

        Args:
            names (list): Optional list of names of contrasts to return. If
                None (default), all contrasts are returned.
            entities (bool): If True, concatenates entity columns to the
                returned contrast matrix.

        Returns:
            A ContrastMatrixInfo namedtuple.
        '''

        # Verify that there are no invalid columns in the condition_lists
        all_conds = [c['condition_list'] for c in self._block_contrasts]
        all_conds = set(chain(*all_conds))
        bad_conds = all_conds - set(self.collection.variables.keys())
        if bad_conds:
            raise ValueError("Invalid condition names passed in one or more "
                             " contrast condition lists: %s." % bad_conds)

        # Construct a list of all contrasts, including identity contrasts
        contrasts = list(self._block_contrasts)

        if self.identity_contrasts:
            for col_name in self.collection.variables.keys():
                contrasts.append({
                    'name': col_name,
                    'condition_list': [col_name],
                    'weights': [1],
                })

        # Filter on desired contrast names if passed
        if names is not None:
            contrasts = [c for c in contrasts if c['name'] in names]

        # Build a "maximal" contrast matrix that has all possible rows and
        # columns. Then we'll proceed by knocking out invalid/missing rows and
        # columns separately for each input node.
        contrast_defs = [pd.Series(c['weights'], index=c['condition_list'])
                         for c in contrasts]
        con_mat = pd.DataFrame(contrast_defs).fillna(0).T
        con_mat.columns = [c['name'] for c in contrasts]

        # Identify all variable names in the current collection that don't show
        # up in any of the input nodes. This will include any variables read in
        # at the current level--e.g., if we're at session level, it might
        # include run-by-run ratings of mood, etc.
        inputs = []
        node_cols = [set(n.contrasts[0].columns) for n in self.input_nodes]
        all_cols = set(chain(*node_cols))
        common_vars = set(self.collection.variables.keys()) - all_cols

        # Also track entities for each row in each input node's contrast matrix
        ent_index = []

        # Loop over input nodes. For each one, get all available columns, and
        # use that to trim down the cloned maximal contrast matrix.
        if self.input_nodes:
            for node in self.input_nodes:
                cols = list(set(node.contrasts[0].columns) | common_vars)
                node_mat = con_mat.copy()
                valid = node_mat.index.isin(cols)
                node_mat = node_mat.loc[valid, :]
                inputs.append(node_mat)
                ent_index.extend([node.entities] * len(node_mat))
        else:
            # If there are no input nodes, we're at run level, so just use
            # the maximal contrast matrix.
            inputs.append(con_mat)

        contrasts = pd.concat(inputs, axis=0)

        # Drop rows that are all zeros
        contrasts = contrasts[(contrasts.T != 0.0).any()]

        index = pd.DataFrame.from_records(ent_index)
        self._contrasts = ContrastMatrixInfo(contrasts, index, self.entities)
        return self._contrasts

    def matches_entities(self, entities, strict=False):
        ''' Determine whether current AnalysisNode matches passed entities.

        Args:
            entities (dict): Dictionary of entities to match. Keys are entity
                names; values are single values or lists.
            strict (bool): If True, _all_ entities in the current Node must
                match in order to return True.
        '''
        return matches_entities(self, entities, strict)


def apply_transformations(collection, transformations, select=None):
    ''' Apply all transformations to the variables in the collection.

    Args:
        transformations (list): List of transformations to apply.
        select (list): Optional list of names of variables to retain after all
            transformations are applied.
    '''
    for t in transformations:
        kwargs = dict(t)
        func = kwargs.pop('name')
        cols = kwargs.pop('input', None)

        if isinstance(func, string_types):
            if func in ('and', 'or'):
                func += '_'
            if not hasattr(transform, func):
                raise ValueError("No transformation '%s' found!" % func)
            func = getattr(transform, func)
            func(collection, cols, **kwargs)

    if select is not None:
        transform.select(collection, select)

    return collection
