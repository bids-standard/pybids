"""BIDS-StatsModels functionality."""

import json
from collections import namedtuple, OrderedDict, Counter, defaultdict
import itertools
from functools import reduce
import re
import fnmatch

import numpy as np
import pandas as pd

from bids.layout import BIDSLayout
from bids.utils import matches_entities, convert_JSON, listify
from bids.variables import (BIDSVariableCollection, merge_collections)
from bids.modeling import transformations as tm
from .model_spec import GLMMSpec, MetaAnalysisSpec
from .report.utils import node_report
import warnings


def validate_model(model):
    """Validate a BIDS-StatsModel structure.

    Parameters
    ----------
    model : dict
        A full BIDS-StatsModels specification document loaded from JSON.

    Returns
    -------
    True if the model passes validation. Raises an exception otherwise.
    """
    # Identify non-unique names
    names = Counter([n['name'] for n in model['nodes']])
    duplicates = [n for n, count in names.items() if count > 1]
    if duplicates:
        raise ValueError("Non-unique node names found: '{}'. Please ensure"
                            " all nodes in the model have unique names."
                            .format(duplicates))

    if 'edges' in model:
        for edge in model['edges']:
            if edge['source'] not in names:
                raise ValueError("Missing source node: '{}'".format(edge['source']))
            if edge['destination'] not in names:
                raise ValueError("Missing destination node: '{}'".format(
                    edge['destination']))

    # XXX: May 2021: Helping old models to work. This shouldn't last more than 2 years.
    for node in model["nodes"]:
        if "type" in node.get("dummy_contrasts", {}):
            warnings.warn(f"[Node {node['name']}]: Contrast 'Type' is now 'Test'.")
            node["dummy_contrasts"]["test"] = node["dummy_contrasts"].pop("type")
        for contrast in node.get("contrasts", []):
            if "type" in contrast:
                warnings.warn(f"[Node {node['name']}; Contrast {contrast['name']}]:"
                              "Contrast 'Type' is now 'Test'.")
                contrast["test"] = contrast.pop("type")
        if isinstance(node.get("transformations"), list):
            transformations = {"transformer": "pybids-transforms-v1",
                               "instructions": node["transformations"]}
            warnings.warn(f"[Node {node['name']}]:"
                          f"Transformations reformatted to {transformations}")
            node["transformations"] = transformations
    return True


BIDSStatsModelsEdge = namedtuple('BIDSStatsModelsEdge',
                                 ('source', 'destination', 'filter'))


ContrastInfo = namedtuple('ContrastInfo', ('name', 'conditions', 'weights',
                                           'test', 'entities'))


class BIDSStatsModelsGraph:
    """Rooted graph structure representing the contents of a BIDS-StatsModel file.

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
        self.model = self._load_model(model)
        self.nodes = self._load_nodes(self.model)
        self.edges = self._load_edges(self.model, self.nodes)
        self._root_node = self.model.get('root', list(self.nodes.values())[0])

    def __repr__(self):
        return f"<{self.__class__.__name__}[{{name='{self.model['name']}', description='{self.model['description']}', ... }}]>"

    def __getitem__(self, key):
        '''Alias for get_node(key).'''
        return self.get_node(key)

    @property
    def root_node(self):
        """Returns the graph's root node."""
        return self._root_node

    @staticmethod
    def _load_model(model, validate=True):
        # Load model info from JSON and do some validation
        if isinstance(model, str):
            with open(model, 'r', encoding='utf-8') as fobj:
                model = json.load(fobj)
        # Convert JSON from CamelCase to snake_case keys
        model = convert_JSON(model)
        if validate:
            validate_model(model)
        return model

    @staticmethod
    def _load_nodes(model):
        nodes = OrderedDict()
        for node_args in model['nodes']:
            node = BIDSStatsModelsNode(**node_args)
            nodes[node.name] = node
        return nodes

    @staticmethod
    def _load_edges(model, nodes):
        """Build edges between nodes."""
        edges = model.get('edges', [])
        # assume we have an optional top-level 'Pipeline' field
        if not edges or model.get('pipeline', False):
            node_vals = list(nodes.values())
            for i in range(1, len(node_vals)):
                edges.append({
                    'source': node_vals[i-1].name,
                    'destination': node_vals[i].name,
                })

        for edge in edges:
            src_node, dst_node = nodes[edge['source']], nodes[edge['destination']]
            edge = BIDSStatsModelsEdge(src_node, dst_node, edge.get('filter', {}))
            src_node.add_child(edge)
            dst_node.add_parent(edge)

        return edges

    def get_node(self, name):
        """Return the named node.

        Parameters
        ----------
        name : str
            The name of the node to retrieve (as defined in the BIDS-StatsModel
            document).

        Returns
        -------
        A BIDSStatsModelsNode instance.
        """
        if name not in self.nodes:
            raise KeyError('There is no node with the name "{}".'.format(name))
        return self.nodes[name]

    def load_collections(self, nodes=None, drop_na=False, **kwargs):
        """Load collections in all nodes.

        Parameters
        ----------
        nodes : list
            Optional list of nodes to set up. Each element must be a string
            giving the (unique) name of the node, Nodes that do not match
            will be silently skipped.
        drop_na : bool
            Boolean indicating whether or not to automatically
            drop events that have a n/a amplitude when reading in data
            from event files.
        """
        # Use inputs from BIDS-StatsModel document, and update with kwargs
        selectors = self.model.get('input', {}).copy()
        selectors.update(kwargs)

        for node in self.nodes.values():

            # Skip any nodes whose names don't match list
            if nodes is not None and node.name not in nodes:
                continue

            # Currently allowing scan_length to be passed as a hack; remove for
            # levels above run.
            node_kwargs = selectors.copy()
            if node.level != 'run':
                node_kwargs.pop('scan_length', None)

            collections = self.layout.get_collections(node.level, drop_na=drop_na,
                                                      **node_kwargs)
            node.add_collections(collections)

    def write_graph(self, dotfilename='graph.dot', format='png', pipe=False):
        """Generates a graphviz dot file and a png file

        Parameters
        ----------

        format: 'png', 'svg'

        """

        from graphviz import Digraph

        dot = Digraph(
                'structs',
                filename=dotfilename,
                node_attr={'shape': 'record'},
                comment=self.model['name'],
                format=format, 
            )

        for node, nobj in self.nodes.items():
            dot.node(node, f"<f0> name: {nobj.name}|<f1> level: {nobj.level}")

        for edge in self.edges:
            dot.edge(edge['source'], edge['destination'])

        if pipe:
            dot = dot.pipe(encoding='utf-8')
        else:
            dot.render()

        return dot

    def run_graph(self, entities=None, **kwargs):
        """Run the entire graph recursively.

        Parameters
        ----------
        entities : dict (optional)
            Optional dictionary of BIDS entities to use when loading collections.
        kwargs : dict
            Optional dictionary of keyword arguments to pass to
            BIDSStatsModelsNode.run()
        """
        if entities is None:
            entities = {}
        
        _run_node_recursive(self.root_node, filters=entities, **kwargs)

def _run_node_recursive(node, inputs=None, filters=None, **kwargs):
    """
    Run a node recursively, storing outputs in place as
    BIDSStatsModelsNode.outputs_.

    """
    if filters == None:
        filters = {}
        
    # Run node
    run_kwargs = {**filters, **kwargs}
    node.outputs_ =  node.run(inputs, group_by=node.group_by, **run_kwargs)

    # Inputs to next node
    contrasts = list(itertools.chain(*[s.contrasts for s in node.outputs_]))

    for edge in node.children:
        _run_node_recursive(edge.destination, contrasts, filters=edge.filter, **kwargs)

class BIDSStatsModelsNode:
    """Represents a single node in a BIDS-StatsModel graph.

    Parameters
    ----------
    level : str
        The BIDS keyword to use as the grouping variable; must be one of
        ['run', 'session', 'subject', or 'dataset'].
    name : str
        Name to assign to the node.
    model : dict
        The 'model' part of the BIDS-StatsModels specification.
    transformations : dict
        Optional dictionary specifying transformations to apply. Dictionary
        must include "transformer" and "instructions" keys. "transformer"
        indicates the specification to follow. "instructions" is a list of
        instructions matching that specification.
    contrasts : list
        List of contrasts to apply to the parameter estimates generated when
        the model is fit.
    dummy_contrasts : dict
        Optional dictionary specifying which conditions to create indicator
        contrasts for. Dictionary may include a "test" key ('t'),
        and optionally a subset of "conditions".
    group_by: [str]
        Optional list of strings giving the names of entities that define the
        grouping structure for all variables. The current node will be executed
        separately for each unique combination of levels specified in group_by.
        For example, if group_by=['contrast', 'subject'], and there are 2
        contrasts and 3 subjects, then there will be 6 separate iterations, and
        the returned list will have 6 elements. Any value passed here will be
        overridden if one is passed when run() is called on a node.
    """

    def __init__(self, level, name, model, group_by, transformations=None,
                 contrasts=None, dummy_contrasts=False):
        self.level = level.lower()
        self.name = name
        self.model = model
        if transformations is None:
            transformations = {"transformer": "pybids-transforms-v1",
                               "instructions": []}
        self.transformations = transformations
        self.contrasts = contrasts or []
        self.dummy_contrasts = dummy_contrasts
        self._collections = []
        self._group_data = []
        self.children = []
        self.parents = []
        if group_by is None:
            raise ValueError(f"group_by is not defined for Node: {name}")
        self.group_by = group_by

        # Check for intercept only run level model and throw an error
        try:
            if (self.level == 'run') and (self.model['x'] == [1]):
                raise NotImplementedError("Run level intercept only models are not currently supported."
                                          "If this is a feature you need, please leave a comment at"
                                          "https://github.com/bids-standard/pybids/issues/852.")
        except KeyError:
            # We talked about X being required, I don't know if we want to throw an error over that requirement here
            # though.
            pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(level={self.level}, name={self.name})>"

    @staticmethod
    def _build_groups(objects, group_by):
        """Group list of objects into bins defined by specified entities.

        Parameters
        ----------
        objects : list
            List of objects containing an .entities dictionary as an attribute.
        group_by : list of str
            List of strings indicating which entities to group on.

        Returns
        -------
        A dictionary, where the keys are tuples of tuples and the values are lists
        containing subsets of the original object list. Each outer tuple of the keys
        represents a particular entity/value combination; each inner tuple has two
        elements, where the first is the name of the entity and the second is the value.

        Notes
        -----
        * The passed objects can be of any type, but alll elements in the list must
        have an implemented .entities dictionary.
        * Objects that do not have defined values for all of the requested
        `entities` will be silently ignored.

        Examples
        --------
        >>> bidsfile = namedtuple('bidsfile', ('entities',))
        >>> bidsfile1 = bidsfile(entities={'subject': '01'})
        >>> bidsfile2 = bidsfile(entities={'subject': '02'})
        >>> groups = {(('subject', '01'),): [bidsfile1],
        ...           (('subject', '02'),): [bidsfile2]}
        >>> BIDSStatsModelsNode._build_groups(
        ...     [bidsfile1, bidsfile2],
        ...     ['subject']) == groups
        True
        """
        if not group_by:
            return {(): objects}

        groups = defaultdict(list)

        # Get unique values in each grouping variable and construct indexing DF
        entities = [obj.entities for obj in objects]
        df = pd.DataFrame.from_records(entities)

        # Separate BIDSVariableCollections
        collections = [obj.to_df() for obj in objects if type(obj) == BIDSVariableCollection]
        if collections:
            metadata_vars = reduce(pd.DataFrame.merge, collections)
            on_vars = list({'subject', 'session', 'run'} & set(metadata_vars.columns))
            df = df.merge(metadata_vars, how='left', on=on_vars)

        # Single-run tasks and single-session subjects may not have entities
        dummy_groups = {"run", "session"} - set(df.columns)
        group_by = set(group_by) - dummy_groups

        # Verify all columns in group_by exist and raise sensible error if not
        missing_vars = list(group_by - set(df.columns))
        if missing_vars:
            raise ValueError("group_by contains variable(s) {} that could not "
                             "be found in the entity index.".format(missing_vars) )

        # Restrict DF to only grouping columns
        df = df.loc[:, list(group_by)]

        unique_vals = {col: df[col].dropna().unique().tolist() for col in group_by}

        # Note: we can't just naively bucket objects based on the values of the
        # grouping entities, because an object may have undefined values for
        # one or more grouping variables. So we "fill in" the missing values by
        # Looping over objects. For each one, identify all grouping variables
        # with missing values. Loop over the cartesian product of the unique
        # values of the missing entities, combining the permutation values with
        # the base (present) grouping entities. Sort on this to produce the
        # group key.
        for i, row in df.iterrows():

            defined = df.columns[df.iloc[i].notnull()].tolist()
            base_ents = [(k, row[k]) for k in defined]
            missing = df.columns[df.iloc[i].isnull()].tolist()

            records = []

            if missing:
                product = itertools.product(*[unique_vals[col] for col in missing])
                for perm in product:
                    fill_ents = [(k, v) for (k, v) in dict(zip(missing, perm)).items()]
                    records.append(base_ents + fill_ents)
            else:
                records.append(base_ents)

            for rec in records:
                grp_key = tuple(sorted(rec))
                groups[grp_key].append(objects[i])

        return groups

    def run(self, inputs=None, group_by=None, force_dense=True,
              sampling_rate='TR', invalid_contrasts='drop', missing_values=None,
              transformation_history=False, node_reports=False, **filters):
        """Execute node with provided inputs.

        Parameters
        ----------
        inputs: [[ContrastInfo]]
            List of list of ContrastInfo tuples passed from the previous node.
            Each element in the outer list maps to the output of a unit at the
            previous level; each element in the inner list is a ContrastInfo
            tuple. E.g., if contrast information is being passed from run-level
            to subject-level, each outer element is a run.
        group_by: [str]
            Optional list of strings giving the names of entities that define
            the grouping structure for all variables. The current node will be
            executed separately for each unique combination of levels specified
            in group_by. For example, if group_by=['contrast', 'subject'], and
            there are 2 contrasts and 3 subjects, then there will be 6 separate
            iterations, and the returned list will have 6 elements. If None is
            passed, the value set at node initialization (if any) will be used.
        force_dense: bool
            If True, the returned design matrices contained in ModelSpec
            instances will represent time in a dense (i.e., uniform temporal
            sampling) format. If False, the returned format will be sparse if
            all available variables are sparse, and dense otherwise. Ignored
            if none of the variables at this node are run-level.
        sampling_rate: str, float
            The sampling rate to use for timeseries resampling when working
            with run-level variables and returning dense output. Ignored if
            there are no run-level variables, or if force_dense is False and
            all available variables are sparse.
        invalid_contrasts: str
            Indicates how to handle invalid contrasts--i.e., ones where the
            specification contains variables that aren't found at run-time.
            Valid values:
                * 'drop' (default): Drop invalid contrasts, retain the rest.
                * 'ignore': Keep invalid contrasts despite the missing variables.
                * 'error': Raise an error.
        missing_values: str
            Indicates how to handle missing values in the data. Valid values:
                * 'fill' (default): Fill missing values with 0.
                * 'ignore': Keep missing values.
                * 'error': Raise an error.
            Defaults to 'fill' if None, but raises warning if not explicitly set.
        transformation_history: bool
            If True, the returned ModelSpec instances will include a history of
            variable collections after each transformation.
        node_reports: bool
            If True, report measures and plots will be included in the output
        filters: dict
            Optional keyword arguments used to constrain the subset of the data
            that's processed. E.g., passing subject='01' will process and
            return data only for files associated with subject '01'.

        Returns
        -------
        A list of BIDSStatsModelsNodeOutput instances.
        """

        inputs = inputs or []
        collections = self._collections
        group_by = listify(group_by or self.group_by)

        # Filter inputs and collections if needed
        if filters:
            inputs = [i for i in inputs if matches_entities(i, filters)]
            collections = [c for c in collections if matches_entities(c, filters)]

        # group all collections and inputs
        all_objects = inputs + collections
        groups = self._build_groups(all_objects, group_by)

        results = []

        for grp_ents, grp_objs in list(groups.items()):

            # split group's objects into inputs and collections
            grp_inputs, grp_colls = [], []
            for obj in grp_objs:
                if isinstance(obj, BIDSVariableCollection):
                    grp_colls.append(obj)
                else:
                    grp_inputs.append(obj)

            node_output = BIDSStatsModelsNodeOutput(
                node=self, entities=dict(grp_ents), collections=grp_colls,
                inputs=grp_inputs, force_dense=force_dense,
                sampling_rate=sampling_rate, invalid_contrasts=invalid_contrasts,
                missing_values=missing_values,
                transformation_history=transformation_history, node_reports=node_reports)
            results.append(node_output)

        return results

    def add_child(self, edge):
        """Add an edge to a child node.

        Parameters
        ----------
        edge : BIDSStatsModelsEdge
            An edge to add to the list of children.
        """
        self.children.append(edge)

    def add_parent(self, edge):
        """Add an edge to a parent node.

        Parameters
        ----------
        edge : BIDSStatsModelsEdge
            An edge to add to the list of children.
        """
        self.parents.append(edge)

    def add_collections(self, collections):
        """Add BIDSVariableCollections (i.e., predictors) to the current node.

        Parameters
        ----------
        collections : [BIDSVariableCollection]
            List of BIDSVariableCollection objects to add to current node.

        Notes
        -----
        No checking for redundancy is performed, so if load_collections() is
        invoked multiple times with overlapping selectors, redundant predictors
        are likely to be stored internally.
        """
        self._collections.extend(collections)

    def get_collections(self, **filters):
        """Returns BIDSVariableCollections at the current node.
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
        # Keeps only collections that match target entities, and also removes
        # those keys from the kwargs dict.
        return [c for c in self._collections if matches_entities(c, filters)]


def expand_wildcards(selectors, pool):
    out = list(selectors)
    for spec in selectors:
        if re.search(r'[\*\?\[\]]', spec):
            idx = out.index(spec)
            out[idx:idx + 1] = fnmatch.filter(pool, spec)
    return out


class BIDSStatsModelsNodeOutput:
    """Represents a single node in a BIDSStatsModelsGraph.

    Parameters
    ----------
    node : BIDSStatsModelsNode
        The node that generated the current instance.
    entities : dict
        Dictionary of entities/metadata applicable to the current node output.
    collections : [BIDSVariableCollection]
        List of BIDSVariableCollection instances needed too generate outputs.
    inputs : [ContrastInfo]
        List of ContrastInfo instances used to generate outputs.
    force_dense: bool
        If True, the returned design matrices contained in ModelSpec instances
        will represent time in a dense (i.e., uniform temporal sampling) format.
        If False, the returned format will be sparse if all available variables
        are sparse, and dense otherwise. Ignored if none of the variables at
        this node are run-level.
    sampling_rate: str, float
        The sampling rate to use for timeseries resampling when working with
        run-level variables and returning dense output. Ignored if there are no
        run-level variables, or if force_dense is False and all available
        variables are sparse.
    invalid_contrasts: str
        Indicates how to handle invalid contrasts--i.e., ones where the
        specification contains variables that aren't found at run-time.
        Valid values:
            * 'drop' (default): Drop invalid contrasts, retain the rest.
            * 'ignore': Keep invalid contrasts despite the missing variables.
            * 'error': Raise an error.
    missing_values: str
        Indicates how to handle missing values in the data. Valid values:
            * 'fill' (default): Fill missing values with 0.
            * 'ignore': Keep missing values.
            * 'error': Raise an error.
        Defaults to 'fill' if None, but raises warning if not explicitly set.
    transformation_history: bool
        If True, the returned ModelSpec instances will include a history of
        variable collections after each transformation.
    node_reports: bool
        If True, a report will be generated for each node output.
    """
    def __init__(self, node, entities={}, collections=None, inputs=None,
                 force_dense=True, sampling_rate='TR', invalid_contrasts='drop',
                 missing_values=None, *, transformation_history=False,
                 node_reports=False):
        """Initialize a new BIDSStatsModelsNodeOutput instance.
        Applies the node's model to the specified collections and inputs, including
        applying transformations and generating final model specs and design matrices (X).
        """
        collections = collections or []
        inputs = inputs or []

        self.node = node
        self.entities = entities
        self.force_dense = force_dense
        self.sampling_rate = sampling_rate
        self.invalid_contrasts = invalid_contrasts
        self.coll_hist = None

        # Apply transformations and convert collections to single DF
        dfs = self._collections_to_dfs(collections, collection_history=transformation_history)

        if inputs:
            dfs.append(self._inputs_to_df(inputs))

        df = reduce(pd.DataFrame.merge, dfs)

        # If dummy coded condition columns needed, generate and concat
        var_names = list(self.node.model['x'])
        if inputs:
            dummy_df = pd.get_dummies(df['contrast'])
            dummies_needed = set(var_names).intersection(dummy_df)
            if dummies_needed:
                df = pd.concat([df, dummy_df[list(dummies_needed)]], axis=1)

        # If a single incoming contrast, keep track of name
        if 'contrast' in df.columns and df['contrast'].nunique() == 1:
            unique_in_contrast = df['contrast'].unique()[0]
        else:
            unique_in_contrast = None

        # Handle the special 1 construct.
        # Add column of 1's to the design matrix called "intercept"
        if 1 in var_names:
            if "intercept" in var_names:
                raise ValueError("Cannot define both '1' and 'intercept' in 'X'")

            var_names = ['intercept' if i == 1 else i for i in var_names]
            if 'intercept' not in df.columns:
                df.insert(0, 'intercept', 1)

        var_names = expand_wildcards(var_names, df.columns)

        # Verify all X names are actually present
        missing = list(set(var_names) - set(df.columns))
        if missing:
            raise ValueError("X specification includes variable(s) {}, but "
                             "these were not found in data matrix.".format(missing))

        # separate the design columns from the entity columns
        self.data = df.loc[:, var_names]
        self.metadata = df.loc[:, df.columns.difference(var_names)]

        # replace missing values with 0 and warn user which columns had missing values
        if missing_values != 'ignore':
            missing_cols = self.data.columns[self.data.isnull().any()]
            if len(missing_cols) > 0:
                base_message = f"NaNs detected in columns: {missing_cols}"

                if missing_values in ['fill', None]:
                    self.data.fillna(0, inplace=True)
                    if missing_values is None:
                        base_message += " were replaced with 0.  Consider "\
                            " handling missing values using transformations."
                        warnings.warn(base_message)
                elif missing_values == 'error':
                    base_message += ". Explicitly replace missing values using transformations."
                    raise ValueError(base_message)

        # create ModelSpec and build contrasts
        kind = node.model.get('type', 'glm').lower()
        SpecCls = {
            'glm': GLMMSpec,
            'meta': MetaAnalysisSpec,
        }[kind]
        self.model_spec = SpecCls.from_df(self.data, node.model, self.metadata)
        self.contrasts = self._build_contrasts(unique_in_contrast)

        self.report_ = node_report(self) if node_reports else None

    def _collections_to_dfs(self, collections, *, collection_history=False):
        """Merges collections and converts them to a pandas DataFrame."""
        if not collections:
            return []

        # group all collections by level
        coll_levels = defaultdict(list)
        [coll_levels[coll.level].append(coll) for coll in collections]

        var_names = list(set(self.node.model['x']) - {1})

        grp_dfs = []
        trans_hist = []
        # merge all collections at each level and export to a DataFrame
        for level, colls in coll_levels.items():

            # Note: we currently merge _before_ selecting variables. Selecting
            # variables first could be done by passing `variables=all_vars` as
            # an argument on the next line), but we can't do this right now
            # because we can't guarantee that all the variables named in `X`
            # in the model section already exist; some might be created by the
            # transformations.
            coll = merge_collections(colls)



            # apply transformations
            transformations = self.node.transformations
            if transformations:
                transformer = tm.TransformerManager(
                    transformations['transformer'], keep_history=collection_history)
                coll = transformer.transform(coll.clone(), transformations['instructions'])

                if hasattr(transformer, 'history_'):
                    trans_hist += transformer.history_

            # Take the intersection of variables and Model.X (var_names), ignoring missing
            # variables (usually contrasts)
            coll.variables = {v: coll.variables[v]
                              for v in expand_wildcards(var_names, coll.variables)
                              if v in coll.variables}
            if not coll.variables:
                continue

            # run collections need to be handled separately because to_df()
            # takes extra arguments related to handling of time
            if level == 'run':
                if self.force_dense:
                    coll = coll.to_dense(sampling_rate=self.sampling_rate)
                coll = coll.to_df(sampling_rate=self.sampling_rate)
            else:
                coll = coll.to_df()
            grp_dfs.append(coll)

        if collection_history:
            self.trans_hist = trans_hist

        return grp_dfs

    def _inputs_to_df(self, inputs):
        # Convert the inputs to a DataFrame and add to list. Each row is
        # an input; each column is an entity
        rows = [{**con.entities, 'contrast': con.name} for con in inputs]
        input_df = pd.DataFrame.from_records(rows)
        return input_df

    def _build_contrasts(self, unique_in_contrast=None):
        """Contrast list of ContrastInfo objects based on current state.

        Parameters
        ----------
        unique_in_contrast : string
            Name of unique incoming contrast inputs (i.e. if there is only 1)
        """
        in_contrasts = self.node.contrasts.copy()
        col_names = set(self.X.columns)

        # Create dummy contrasts as regular contrasts
        dummies = self.node.dummy_contrasts
        if dummies:
            if {'conditions', 'condition_list'} & set(dummies):
                warnings.warn(
                    "Use 'Contrasts' not 'Conditions' or 'ConditionList' to specify"
                    "DummyContrasts. Renaming to 'Contrasts' for now.")
                dummies['contrasts'] = dummies.pop('conditions', None) or dummies.pop('condition_list', None)

            if 'contrasts' in dummies:
                conditions = set(dummies['contrasts'])
            else:
                conditions = col_names

            for col_name in conditions:
                if col_name == "intercept":
                    col_name = 1

                in_contrasts.insert(0,
                    {
                        'name': col_name,
                        'condition_list': [col_name],
                        'weights': [1],
                        'test': dummies.get('test')
                    }
                )

        # Process all contrasts, starting with dummy contrasts
        # Dummy contrasts are replaced if a contrast is defined with same name
        contrasts = {}
        for con in in_contrasts:
            condition_list = list(con["condition_list"])

            # Rename special 1 construct
            condition_list = ['intercept' if i == 1 else i for i in condition_list]

            name = con["name"]

            # Rename contrast name
            if name == 1:
                name = unique_in_contrast or 'intercept'
            else:
                # If Node has single contrast input, as is grouped by contrast
                # Rename contrast to append incoming contrast name
                if unique_in_contrast:
                    name = f"{unique_in_contrast}_{name}"

            missing_vars = set(condition_list) - col_names
            if missing_vars:
                if self.invalid_contrasts == 'error':
                    msg = ("Variable(s) '{}' specified in condition list for "
                           "contrast '{}', but not found in available data."
                           .format(missing_vars, con['name']))
                    raise ValueError(msg)
                elif self.invalid_contrasts == 'drop':
                    continue
            weights = np.atleast_2d(con['weights'])

            # Add contrast name to entities; can be used in grouping downstream
            entities = {**self.entities, 'contrast': name}
            ci = ContrastInfo(name, condition_list,
                              con['weights'], con.get("test"), entities)
            contrasts[name] = ci

        return list(contrasts.values())

    @property
    def X(self):
        """Return design matrix via the current ModelSpec."""
        return self.model_spec.X

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.node.name}, entities={self.entities})>"
