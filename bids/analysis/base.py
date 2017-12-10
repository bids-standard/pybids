import json
from .variables import load_variables, SimpleColumn, BIDSVariableCollection
from . import transform
from collections import namedtuple
from six import string_types
import pandas as pd
import numpy as np


DesignMatrix = namedtuple('DesignMatrix', ('data', 'image', 'entities'))


class Analysis(object):

    ''' Represents an entire BIDS-Model analysis.
    Args:
        layouts (BIDSLayout or list): One or more BIDSLayout objects to pull
            variables from.
        model (str or dict): a BIDS model specification. Can either be a
            string giving the path of the JSON model spec, or an already-loaded
            dict containing the model info.
        collection (BIDSVariableCollection): Optional BIDSVariableCollection
            object used to load/manage variables. If None, a new collection is
            initialized from the provided layouts.
        selectors (dict): Optional keyword arguments to pass onto the
            collection; these will be passed on to the Layout's .get() method,
            and can be used to restrict variables.
    '''

    def __init__(self, model, collection=None, layouts=None, **selectors):

        if layouts is None and collection is None:
            raise ValueError("At least one of the 'layout' and 'collection' "
                             "arguments must be passed.")

        if isinstance(model, str):
            model = json.load(open(model))
        self.model = model

        if 'input' in model:
            selectors.update(model['input'])

        if collection is None:
            collection = load_variables(layouts, **selectors)

        self.collection = collection

        self._load_blocks(model['blocks'])

    @property
    def layout(self):
        return self.collection.layout  # for convenience

    def __iter__(self):
        for b in self.blocks:
            yield b

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.blocks[index]
        return list(filter(lambda x: x.name == index, self.blocks))[0]

    def _load_blocks(self, blocks):
        self.blocks = []
        for i, b in enumerate(blocks):
            self.blocks.append(Block(self, index=i, **b))

    def setup(self, apply_transformations=True):
        ''' Set up the sequence of blocks--applying transformations,
        generating outputs, etc. '''

        # pass a copy of the collection through the pipeline (columns mutate)
        _collection = self.collection.clone()

        for b in self.blocks:
            b.setup(_collection, apply_transformations=apply_transformations)
            _collection = b.output_collection.clone()


class Block(object):

    ''' Represents a single analysis block from a BIDS-Model specification.
    Args:
        analysis (Analysis): The parent Analysis this Block belongs to.
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

    def __init__(self, analysis, level, index, name=None, transformations=None,
                 model=None, contrasts=None):

        self.analysis = analysis
        self.level = level
        self.index = index
        self.name = name
        self.transformations = transformations or []
        self.model = model or None
        self.contrasts = contrasts or []
        self._design_matrix = None
        self.input_collection = None
        self.output_collection = None

    def _get_design_matrix(self, **selectors):
        if self._design_matrix is None:
            raise ValueError("Block hasn't been set up yet; please call "
                             "setup() before you try to retrieve the DM.")
        # subset the data if needed
        data = self._design_matrix
        if selectors:
            # TODO: make sure this handles constraints on int columns properly
            bad_keys = list(set(selectors.keys()) - set(data.columns))
            if bad_keys:
                raise ValueError("The following query constraints do not map "
                                 "onto existing columns: %s." % bad_keys)
            query = ' and '.join(["{} in {}".format(k, v)
                                  for k, v in selectors.items()])
            data = data.query(query)
        return data

    def _drop_columns(self, data, drop_entities=True, drop_timing=False):

        entities = []
        if drop_entities:
            entities += ['run', 'session', 'subject', 'task']
        if drop_timing:
            entities += ['onset', 'duration']
        common_ents = list(set(entities) & set(data.columns))
        return data.drop(common_ents, axis=1)

    def _get_groupby_cols(self):
        # Get a list of keywords that define the grouping at the current level.
        # Note that we need to include all entities *above* the current one--
        # e.g., if the block level is 'run', this means we actually want to
        # groupby(['subject', 'session', 'run']), otherwise we would only
        # end up with n(runs) groups.
        hierarchy = ['subject', 'session', 'run']

        if self.level is None:
            return None
        elif self.level not in hierarchy:  # e.g., for 'dataset'
            return []
        pos = hierarchy.index(self.level)
        return hierarchy[:(pos + 1)]

    def apply_transformations(self):
        ''' Apply all transformations to the variables in the collection.
        '''
        for t in self.transformations:
            kwargs = dict(t)
            func = kwargs.pop('name')
            cols = kwargs.pop('input', None)

            if isinstance(func, string_types):
                if not hasattr(transform, func):
                    raise ValueError("No transformation '%s' found!" % func)
                func = getattr(transform, func)
                func(self.input_collection, cols, **kwargs)

        # Also apply variable selection even if it's not represented in
        # its own transformation.
        if self.model is not None and 'variables' in self.model:
            # TODO: add a "select" transformation and construct a call to it
            # here instead of filtering directly on the collection.
            coll = self.input_collection
            coll.columns = {k: v for k, v in coll.columns.items()
                            if k in self.model['variables']}

    def generate_output_collection(self, keep_input_columns=True):

        data = self._get_design_matrix()

        # Figure out which column names to keep
        contrast_names = [c['name'] for c in self.contrasts]
        if keep_input_columns:
            sel_cols = list(data['condition'].unique())
            contrast_names = sel_cols + contrast_names

        ent_cols = self._get_groupby_cols()
        ent_cols = list(set(ent_cols) & set(data.columns))

        contrast_names = list(set(contrast_names) -
                              set(self.input_collection.entities))

        # Set the same (unit) weights for all rows
        unit_weights = pd.Series(np.ones(len(contrast_names)),
                                 index=contrast_names)

        if ent_cols:
            data = data.groupby(ent_cols).apply(lambda x: unit_weights)

        # Generate a new BIDSVariableCollection to pass to next block
        ic = self.input_collection

        collection = BIDSVariableCollection(ic.layout, entities=ent_cols)

        for col_name in data.columns:
            col_data = data[col_name].reset_index()
            col_data = col_data.rename(columns={col_name: 'amplitude'})
            col = SimpleColumn(collection, col_name, col_data)
            collection.columns[col_name] = col

        self.output_collection = collection

    def get_contrasts(self, format='matrix', **selectors):
        ''' Return contrast information for the current block.
        Args:
            format (str): What format to return the contrast specifications in.
                Valid options are:
                    'matrix' or 'df': Returns a pandas DataFrame with each
                        contrast as a row and each of the existing design
                        matrix columns as a column. This format makes it easy
                        to matrix-multiply existing in-memory images by the
                        contrast definition matrix in one shot.
                    'patsy': Returns a list of strings, where each string gives
                        a patsy-compatible definition of the contrast. E.g.,
                        if there are conditions 'A' and 'B', and the weights
                        are [1, -1], the returned string would be "A-B".
                    'dict': Returns the BIDS-Model contrast specification
                        as a dict loaded from the original json.
                    'json': Returns the json string containing the raw contrast
                        specification found in the original BIDS-Model spec.
            selectors (dict): Optional keyword arguments to further constrain
                the data retrieved.
        Returns:
            See format argument for returned object formats.
        '''
        if format == 'dict':
            return self.contrasts

        if format == 'json':
            return json.dumps(self.contrasts)

        # Construct contrast x variable matrix
        pass

    def get_Xy(self, drop_entities=True, **selectors):
        ''' Return X and y information for all groups defined by the current
        level.
        Args:
            selectors (dict): Optional keyword arguments to further constrain
                the data retrieved.
        Returns:
            A list of triples, where each triple contains data for a single
                group, and the elements reflect (in order):
                    - The design matrix containing all of the predictors (X);
                    - The filename of the 4D image associated with the
                      current group/design matrix;
                    - A dict of entities defining the current group (e.g.,
                      {'subject': '03', 'run': 1})
        '''
        data = self._get_design_matrix(**selectors)
        ent_cols = self._get_groupby_cols()

        tuples = []
        ent_cols = list(set(ent_cols) & set(data.columns))

        # If there are no entities to group on, return the whole dataset
        if not ent_cols:
            return [(data, None, {})]

        # Otherwise loop over groups and construct a tuple for each one
        for name, g in data.groupby(ent_cols):
            ent_data = g[ent_cols].drop_duplicates().iloc[0, :]
            ents = ent_data.to_dict()
            if 'run' in ent_cols:
                img = self.analysis.layout.get(return_type='file', type='bold',
                                               modality='func',
                                               extensions='.nii.gz', **ents)
                img = img[0]
            else:
                img = None
            group_data = self._drop_columns(g.copy(),
                                            drop_entities=drop_entities)
            record = DesignMatrix(group_data, img, ents)
            tuples.append(record)
        return tuples

    def iter_Xy(self, **selectors):
        ''' Convenience method that returns an iterator over tuples returned
        by get_Xy(). See get_Xy() for arguments and return format. '''
        return (t for t in self.get_Xy(**selectors))

    def setup(self, collection, apply_transformations=True,
              generate_output_collection=True):
        ''' Set up the Block and construct the design matrix.
        Args:
            collection (BIDSVariableCollection): The variable collection to
                use. Note that the setup process will often mutate the
                collection instance.
            last_level (str): The level of the previous Block in the analysis,
                if any.
            apply_transformations (bool): If True (default), apply any
                transformations in the block before constructing the design
                matrix.
            generate_output_collection (bool): If True (default), generate the
                output collection to pass to the next block as its input.
        '''
        self.input_collection = collection
        self.output_collection = None

        if self.transformations and apply_transformations:
            self.apply_transformations()

        gb = self._get_groupby_cols()

        self._design_matrix = collection.get_design_matrix(groupby=gb).copy()

        if self.generate_output_collection:
            self.generate_output_collection()
