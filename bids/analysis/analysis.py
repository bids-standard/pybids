import json
from bids.grabbids import BIDSLayout
from bids.utils import matches_entities
from . import transformations as transform
from collections import namedtuple
from six import string_types
import pandas as pd


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
    '''

    def __init__(self, layout, model, collections=None, sampling_rate=None,
                 scan_length=None, **selectors):

        self.sampling_rate = sampling_rate
        self.collections = collections
        self.selectors = selectors
        self.scan_length = scan_length

        if not isinstance(layout, BIDSLayout):
            layout = BIDSLayout(layout)
        self.layout = layout

        if isinstance(model, str):
            model = json.load(open(model))
        self.model = model

        if 'input' in model:
            selectors.update(model['input'])

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

    def setup(self, blocks=None, agg_func='mean', **kwargs):
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
                or a DataFrame.
        '''

        # In the beginning, there was nothing
        input_nodes = None

        for i, b in enumerate(self.blocks):

            # Skip any blocks whose names or indexes don't match block list
            if blocks is not None and i not in blocks and b.name not in blocks:
                continue

            b.setup(input_nodes, **kwargs)
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

    def _filter_objects_by_entities(self, obj, kwargs):
        valid_ents = {'run', 'session', 'subject', 'task'}
        entities = {k: v for k, v in kwargs.items() if k in valid_ents}
        return [o for o in obj if o.matches_entities(entities)]

    def setup(self, input_nodes=None, **kwargs):
        ''' Set up the Block and construct the design matrix.

        Args:
            input_nodes (list): Optional list of Node objects produced by
                the preceding Block in the analysis. If None, uses any inputs
                passed in at Block initialization.
            kwargs: Optional keyword arguments to pass onto load_variables.
        '''

        input_nodes = input_nodes or self.input_nodes
        collections = self.layout.get_collections(self.level, **kwargs)

        collections = self._filter_objects_by_entities(collections, kwargs)

        for coll in collections:
            coll = apply_transformations(coll, self.transformations)
            node = AnalysisNode(self.level, coll, self.contrasts)
            self.output_nodes.append(node)

    def get_design_matrix(self, variables=None, format='long', **kwargs):
        nodes = self._filter_objects_by_entities(self.output_nodes, kwargs)
        return [n.get_design_matrix(variables, format, **kwargs)
                for n in nodes]

    def get_contrasts(self, names=None, identity_contrasts=True, **kwargs):
        nodes = self._filter_objects_by_entities(self.output_nodes, kwargs)
        return [n.get_contrasts(names, identity_contrasts) for n in nodes]


DesignMatrixInfo = namedtuple('DesignMatrixInfo', ('data', 'entities'))
ContrastMatrixInfo = namedtuple('ContrastMatrixInfo', ('data', 'entities'))


class AnalysisNode(object):

    def __init__(self, level, collection, contrasts=None):
        self.level = level
        self.collection = collection
        self.contrasts = contrasts

    @property
    def entities(self):
        return self.collection.entities

    def get_design_matrix(self, variables=None, format='long', **kwargs):
        df = self.collection.to_df(variables, format, **kwargs)
        return DesignMatrixInfo(df, self.entities)

    def get_contrasts(self, names=None, identity_contrasts=True):
        ''' Return contrast information for the current block.

        Args:
            names (list): Optional list of names of contrasts to return. If
                None (default), all contrasts are returned.
            identity_contrasts (bool): If True, all columns in the output
                design matrix are automatically assigned identity contrasts.
                If None, falls back on the identity_contrasts flag defined at
                the block level (which defaults to True).
        '''
        contrasts = self.contrasts.copy()

        if names is not None:
            contrasts = [c for c in contrasts if c['name'] in names]

        if identity_contrasts:
            for col_name in self.collection.variables.keys():
                contrasts.append({
                    'name': col_name,
                    'condition_list': [col_name],
                    'weights': [1],
                })

        contrast_defs = [pd.Series(c['weights'], index=c['condition_list'])
                         for c in contrasts]
        df = pd.DataFrame(contrast_defs).fillna(0)
        df.index = [c['name'] for c in contrasts]
        return ContrastMatrixInfo(df, self.entities)

    def matches_entities(self, entities, strict=False):
        return matches_entities(self, entities, strict)


def apply_transformations(collection, transformations, select=None):
    ''' Apply all transformations to the variables in the collection.
    '''
    for t in transformations:
        kwargs = dict(t)
        func = kwargs.pop('name')
        cols = kwargs.pop('input', None)

        if isinstance(func, string_types):
            if not hasattr(transform, func):
                raise ValueError("No transformation '%s' found!" % func)
            func = getattr(transform, func)
            func(collection, cols, **kwargs)

    if select is not None:
        transform.select(collection, select)

    return collection
