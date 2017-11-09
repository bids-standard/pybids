import json
from .variables import BIDSVariableManager
from collections import namedtuple

DesignMatrix = namedtuple('DesignMatrix', ('entities', 'groupby', 'data'))


class Analysis(object):

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

        self.layout = manager.layout # for convenience

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

    def setup(self):
        self.manager.load()
        # pass the manager through the pipeline
        last_level = None
        for b in self.blocks:
            b.setup(self.manager, last_level)
            last_level = b.level


class Block(object):

    def __init__(self, analysis, level, index, name, transformations=None,
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
        if level is None:
            return None
        hierarchy = ['subject', 'session', 'run']
        pos = hierarchy.index(level)
        return hierarchy[:(pos + 1)]

    def get_Xy(self, **selectors):

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
        return (t for t in self.get_Xy(**selectors))

    def setup(self, manager, last_level=None, input_design_matrix=None):
        agg = 'mean' if self.level != 'run' else None
        last_level = self._get_groupby_cols(last_level)
        self.design_matrix = manager.get_design_matrix(groupby=last_level,
                                                       aggregate=agg).copy()
