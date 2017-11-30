import json
from .variables import BIDSVariableManager
from . import transform
from collections import namedtuple
from six import string_types


DesignMatrix = namedtuple('DesignMatrix', ('entities', 'groupby', 'data'))


class Analysis(object):

    ''' Represents an entire BIDS-Model analysis.
    Args:
        layouts (BIDSLayout or list): One or more BIDSLayout objects to pull
            variables from.
        model (str or dict): a BIDS model specification. Can either be a
            string giving the path of the JSON model spec, or an already-loaded
            dict containing the model info.
        manager (BIDSVariableManager): Optional BIDSVariableManager object used
            to load/manage variables. If None, a new manager is initialized
            from the provided layouts.
        selectors (dict): Optional keyword arguments to pass onto the manager;
            these will be passed on to the Layout's .get() method, and can be
            used to restrict variables.
    '''

    def __init__(self, layouts, model, manager=None, **selectors):
        if isinstance(model, str):
            model = json.load(open(model))
        self.model = model

        if 'input' in model:
            selectors.update(model['input'])

        if manager is None:
            manager = BIDSVariableManager(layouts, **selectors)

        self.manager = manager

        self._load_blocks(model['blocks'])

    @property
    def layout(self):
        return self.manager.layout  # for convenience

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

        ''' Read in all variables and set up the sequence of blocks. '''
        self.manager.load()

        # pass a copy of the manager through the pipeline (columns mutate)
        _manager = self.manager.clone()
        last_level = None

        for b in self.blocks:
            b.setup(_manager, last_level,
                    apply_transformations=apply_transformations)
            last_level = b.level


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
        self.design_matrix = None

    def _get_design_matrix(self, **selectors):
        if self.design_matrix is None:
            raise ValueError("Block hasn't been set up yet; please call "
                             "setup() before you try to retrieve the DM.")
        # subset the data if needed
        data = self.design_matrix
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

    def get_contrast_matrix(self):
        pass

    def _drop_columns(self, data):
        entities = {'onset', 'duration', 'run', 'session', 'subject', 'task'}
        common_ents = list(entities & set(data.columns))
        return data.drop(common_ents, axis=1)

    def _get_groupby_cols(self, level):
        # Get a list of keywords that define the grouping at the current level.
        # Note that we need to include all entities *above* the current one--
        # e.g., if the block level is 'run', this means we actually want to
        # groupby(['subject', 'session', 'run']), otherwise we would only
        # end up with n(runs) groups.
        if level is None:
            return None
        hierarchy = ['subject', 'session', 'run']
        pos = hierarchy.index(level)
        return hierarchy[:(pos + 1)]

    def apply_transformations(self):
        ''' Apply all transformations to the variables in the manager.
        '''
        for t in self.transformations:
            kwargs = dict(t)
            func = kwargs.pop('name')
            cols = kwargs.pop('input', None)

            if isinstance(func, string_types):
                if not hasattr(transform, func):
                    raise ValueError("No transformation '%s' found!" % func)
                func = getattr(transform, func)
                func(self.manager, cols, **kwargs)

    def get_Xy(self, **selectors):
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
        ent_cols = self._get_groupby_cols(self.level)

        tuples = []
        ent_cols = list(set(ent_cols) & set(data.columns))
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
            tuples.append((self._drop_columns(g.copy()), img, ents))
        return tuples

    def iter_Xy(self, **selectors):
        ''' Convenience method that returns an iterator over tuples returned
        by get_Xy(). See get_Xy() for arguments and return format. '''
        return (t for t in self.get_Xy(**selectors))

    def setup(self, manager, last_level=None, apply_transformations=True):
        ''' Set up the Block and construct the design matrix.
        Args:
            manager (BIDSVariableManager): The variable manager to use. Note:
                that the setup process will often mutate the manager instance.
            last_level (str): The level of the previous Block in the analysis,
                if any.
            apply_transformations (bool): If True (default), apply any
                transformations in the block before constructing the design
                matrix.
        '''
        self.manager = manager

        agg = 'mean' if self.level != 'run' else None
        last_level = self._get_groupby_cols(last_level)

        if self.transformations and apply_transformations:
            self.apply_transformations()

        self.design_matrix = manager.get_design_matrix(groupby=last_level,
                                                       aggregate=agg).copy()
