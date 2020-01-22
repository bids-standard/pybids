"""BIDS-StatsModels functionality."""

import json
from collections import namedtuple, OrderedDict
from itertools import chain

import numpy as np
import pandas as pd

from bids.layout import BIDSLayout
from bids.utils import matches_entities, convert_JSON
from bids.variables import (BIDSVariableCollection, SparseRunVariable,
                            merge_collections)
from bids.analysis import transformations as tm
from .model_spec import create_model_spec


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
            raise KeyError('There is no step with the name "%s".' % index)
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
            step = Step(self.layout, index=i, **step_args)
            self.steps.append(step)

    def setup(self, steps=None, drop_na=False, **kwargs):
        """Set up the sequence of steps for analysis.

        Parameters
        ----------
        steps : list
            Optional list of steps to set up. Each element
            must be either an int giving the index of the step in the
            JSON config step list, or a str giving the (unique) name of
            the step, as specified in the JSON config. Steps that do not
            match either index or name will be skipped.
        drop_na : bool
            Boolean indicating whether or not to automatically
            drop events that have a n/a amplitude when reading in data
            from event files.
        """

        # The first Step in the sequence can't have any contrast inputs
        input_contrasts = None

        # Use inputs from model, and update with kwargs
        selectors = self.model.get('input', {}).copy()
        selectors.update(kwargs)

        for i, step in enumerate(self.steps):

            # Skip any steps whose names or indexes don't match step list
            if steps is not None and i not in steps and step.name not in steps:
                continue

            step.setup(input_contrasts, drop_na=drop_na, **selectors)
            input_contrasts = [step.get_contrasts(c)
                               for c in step.get_collections(**selectors)]
            input_contrasts = list(chain(*input_contrasts))


class Step(object):
    """Represents a single analysis step from a BIDS-Model specification.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        The BIDSLayout containing all project files.
    level : str
        The BIDS keyword to use as the grouping variable; must be one of
        ['run', 'session', 'subject', or 'dataset'].
    index : int
        The numerical index of the current Step within the sequence of steps.
    name : str
        Optional name to assign to the step. Must be specified in order to
        enable name-based indexing in the parent Analysis.
    transformations : list
        List of BIDS-Model transformations to apply.
    model : dict
        The 'model' part of the BIDS-StatsModels specification.
    contrasts : list
        List of contrasts to apply to the parameter estimates generated when
        the model is fit.
    inputs : list
        Optional list of BIDSVariableCollections to use as input to this Step
        (typically, the outputs from the preceding Step).
    dummy_contrasts : dict
        Optional dictionary specifying which conditions to create indicator
        contrasts for. Dictionary must include a "type" key ('t' or 'FEMA'),
        and optionally a subset of "conditions". This parameter is over-written
        by the setting in setup() if the latter is passed.
    """

    def __init__(self, layout, level, index, name=None, transformations=None,
                 model=None, contrasts=None, inputs=None, dummy_contrasts=False):
        self.layout = layout
        self.level = level.lower()
        self.index = index
        self.name = name
        self.transformations = transformations or []
        self.model = model or None
        self.contrasts = contrasts or []
        self.inputs = inputs or []
        self.dummy_contrasts = dummy_contrasts
        self._collections = []

    def _filter_collections(self, collections, kwargs):
        # Keeps only collections that match target entities, and also removes
        # those keys from the kwargs dict.
        valid_ents = {'task', 'subject', 'session', 'run'}
        entities = {k: kwargs.pop(k) for k in dict(kwargs) if k in valid_ents}
        collections = [c for c in collections if matches_entities(c, entities)]
        return (collections, kwargs)

    def _group_objects_by_entities(self, objects):
        # Group list of objects into bins defined by all entities at current
        # Step level or higher. E.g., if the level is 'subject', the
        # returned list will have one element per subject, where each element
        # is a list containing all objects that belongs to that subject. Any
        # object with a defined .entities attribute is groupable.
        if self.level == 'dataset':
            return {'dataset': objects}
        groups = OrderedDict()
        valid_ents = ['subject', 'session', 'task', 'run']
        valid_ents = valid_ents[:(valid_ents.index(self.level) + 1)]
        for o in objects:
            key = {k: v for k, v in o.entities.items() if k in valid_ents}
            key = tuple(sorted(key.items(), key=str))
            if key not in groups:
                groups[key] = []
            groups[key].append(o)
        return groups

    def _merge_contrast_inputs(self, inputs):
        """ Merges a list of ContrastInfo tuples and constructs a dict mapping
        from units of the current level to BIDSVariableCollections.

        Parameters
        ----------
        inputs: [[ContrastInfo]]
            List of list of ContrastInfo tuples passed from the previous Step.
            Each element in the outer list maps to the output of a unit at the
            previous level; each element in the inner list is a ContrastInfo
            tuple. E.g., if contrast information is being passed from run-level
            to subject-level, each outer element is a run.

        Returns
        -------
        A dictionary, where the keys are the values of the entities at the
        current level (e.g., '01', '02'...) and the values are
        BIDSVariableCollection containing contrast information.

        Notes
        -----
        Each output BIDSVariableCollection contains information for a single
        unit at the present level. The variables in the collection reflect the
        union of all contrasts found in one or more of the inputs. A value of
        1 indicates that the contrast is present for a given row in the input;
        0 indicates that the contrast was missing.
        """

        groups = self._group_objects_by_entities(inputs)

        ent_cols = list(list(groups.values())[0][0].entities.keys())

        collections = {}

        for name, contrasts in groups.items():
            # Create a DF with contrasts in rows and contrast names in columns
            data = [{**c.entities, **{c.name: 1}} for c in contrasts]
            data = pd.DataFrame.from_records(data)
            # Group by all entities and sum, collapsing over rows belonging
            # to the current unit
            data = data.groupby(ent_cols).sum()
            # Split the DF up into separate data and entities DFs
            entities = data.index.to_frame(index=False)
            data = data.reset_index(drop=True)
            # Construct the collection
            coll = BIDSVariableCollection.from_df(data, entities, self.level)
            collections[name] = coll

        return collections

    def setup(self, inputs=None, drop_na=False, **kwargs):
        """Set up the Step and construct the design matrix.

        Parameters
        ----------
        inputs : list
            Optional list of BIDSVariableCollections produced as output by the
            preceding Step in the analysis. If None, uses inputs passed at
            initialization (if any).
        drop_na : bool
            Boolean indicating whether or not to automatically drop events that
            have a n/a amplitude when reading in data from event files.
        kwargs : dict
            Optional keyword arguments to pass onto load_variables.
        """
        self._collections = []

        # Convert input contrasts to a list of BIDSVariableCollections
        inputs = inputs or self.inputs or []
        input_grps = self._merge_contrast_inputs(inputs) if inputs else {}

        # TODO: remove the scan_length argument entirely once we switch tests
        # to use the synthetic dataset with image headers.
        if self.level != 'run':
            kwargs = kwargs.copy()
            kwargs.pop('scan_length', None)

        # Now handle variables read from the BIDS dataset: read them in, filter
        # on passed selectors, and group by unit of current level
        collections = self.layout.get_collections(self.level, drop_na=drop_na,
                                                  **kwargs)
        collections, _ = self._filter_collections(collections, kwargs)
        groups = self._group_objects_by_entities(collections)

        # Merge in the inputs
        for key, input_ in input_grps.items():
            if key not in groups:
                groups[key] = []
            groups[key].append(input_)

        # Set up and validate variable lists
        model = self.model or {}
        X = model.get('x', [])

        for grp, colls in groups.items():
            coll = merge_collections(colls)

            colls = tm.TransformerManager().transform(coll, self.transformations)

            if X:
                tm.Select(coll, X)

            self._collections.append(coll)

    def get_collections(self, **filters):
        """Returns BIDSVariableCollections at the current Step.

        Parameters
        ----------
        filters : dict
            Optional keyword filters used to constrain which of the available
            collections get returned (e.g., passing subject=['01', '02'] will
            return collections for only subjects '01' and '02').

        Returns
        -------
        list of BIDSVariableCollection instances
            One instance per unit of the current analysis level (e.g., if
            level='run', each element in the list represents the collection
            for a single run).
        """
        return self._filter_collections(self._collections, filters)[0]

    def get_contrasts(self, collection, names=None, variables=None):
        """Return contrast information at this step for the passed collection.

        Parameters
        ----------
        collection : BIDSVariableCollection
            The collection to generate/validate contrasts for.
        names : list
            Optional list of names of contrasts to return. If None (default),
            all contrasts are returned.
        variables : bool
            Optional list of strings giving the names of design matrix columns
            to use when generating the matrix of weights.

        Returns
        -------
        list
            A list of ContrastInfo namedtuples, one per contrast.

        Notes
        -----
        The 'variables' argument take precedence over the natural process
        of column selection. I.e., if a variable shows up in a contrast, but
        isn't named in variables, it will *not* be included in the result.
        """

        # Verify that there are no invalid columns in the condition_lists
        all_conds = [c['condition_list'] for c in self.contrasts]
        all_conds = set(chain(*all_conds))
        bad_conds = all_conds - set(collection.variables.keys())
        if bad_conds:
            raise ValueError("Invalid condition names passed in one or more "
                             " contrast condition lists: %s." % bad_conds)

        # Construct a list of all contrasts, including dummy contrasts
        contrasts = list(self.contrasts)

        # Check that all contrasts have unique name
        contrast_names = [c['name'] for c in contrasts]
        if len(set(contrast_names)) < len(contrast_names):
            raise ValueError("One or more contrasts have the same name")
        contrast_names = list(set(contrast_names))

        if self.dummy_contrasts:
            if 'conditions' in self.dummy_contrasts:
                conditions = [c for c in self.dummy_contrasts['conditions']
                              if c in collection.variables.keys()]
            else:
                conditions = collection.variables.keys()

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

            return ContrastInfo(c['name'], weights, test_type,
                                collection.entities)

        return [setup_contrast(c) for c in contrasts]

    def get_model_spec(self, collection, sampling_rate='TR'):
        """Get a ModelSpec instance for the passed collection.

        Parameters
        ----------
        collection : BIDSVariableCollection
            The BIDSVariableCollection to construct a model for.
        sampling_rate : {'TR', 'highest'} or float
            For run-level models, the sampling rate at which to generate the
            design matrix. When 'TR', the repetition time is used, if
            available, to select the sampling rate (1/TR). When 'highest', all
            variables are resampled to the highest sampling rate of any
            variable in the collection. The sampling rate may also be specified
            explicitly in Hz. Has no effect on non-run-level collections.

        Returns
        -------
        A bids.analysis.model_spec.ModelSpec instance.

        Notes
        -----
        If the passed BIDSVariableCollection contains any sparse variables,
        they will be automatically converted to dense (using the specified
        sampling rate) before the ModelSpec is constructed. For non-run-level
        collections, timing is irrelevant, so the design matrix is constructed
        based on the "as-is" values found in each variable.
        """
        if self.model is None:
            raise ValueError("Cannot generate a ModelSpec instance; no "
                             "BIDS-StatsModels model specification found "
                             "for this step!")

        if collection.level == 'run':
            collection = collection.resample(sampling_rate, force_dense=True)
        return create_model_spec(collection, self.model)


ContrastInfo = namedtuple('ContrastInfo', ('name', 'weights', 'type',
                                           'entities'))
