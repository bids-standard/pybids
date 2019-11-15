import pandas as pd
import numpy as np

from bids.analysis import transformations as pbt
from bids.variables import BIDSRunVariableCollection


class TransformerManager(object):

    def __init__(self, default=None):
        self.transformations = {}
        if default is None:
            # Default to PyBIDS transformations
            default = pbt
        self.default = default

    def _sanitize_name(self, name):
        """ Replace any invalid/reserved transformation names with acceptable
        equivalents. """
        if name in ('And', 'Or'):
            name += '_'
        return name

    def register(self, name, func):
        name = self._sanitize_name(name)
        self.transformations[name] = func

    def transform(self, collection, transformations):
        """Apply all transformations to the variables in the collection.

        Parameters
        ----------
        collection: BIDSVariableCollection
            The BIDSVariableCollection containing variables to transform.
        transformations : list
            List of transformations to apply.
        """
        for t in transformations:
            kwargs = dict(t)
            name = self._sanitize_name(kwargs.pop('Name'))
            cols = kwargs.pop('Input', None)

            # Check registered transformations; fall back on default module
            func = self.transformations.get(name, None)
            if func is None:
                if not hasattr(self.default, name):
                    raise ValueError("No transformation '%s' found: either "
                                     "explicitly register a handler, or pass a"
                                     " default module that supports it." % name)
                func = getattr(self.default, name)
                func(collection, cols, **kwargs)
        return collection


class GLMMSpec:

    def __init__(self, terms=None, X=None, Z=None, groups=None, sigma=None,
                 family=None, link=None, priors=None):
        self.terms = {}
        self.family = family
        self.link = link
        self.sigma = sigma

        if priors is not None:
            self.set_priors(priors)

        if terms is not None:
            for t in terms:
                self.add_term(t)
        if X is not None:
            self.build_fixed_terms(X)
        if Z is not None:
            self.build_var_comps(Z, groups, sigma)

    def set_priors(self, fixed=None, random=None):
        pass

    def build_fixed_terms(self, X):
        """Build one or more fixed terms from the columns of a pandas DF."""
        for col in X.columns:
            data = X.loc[:, col]
            cat = data.dtype.name in ('str', 'category', 'object')
            # TODO: get default prior
            t = Term(col, data, categorical=cat)
            self.add_term(t)

    def build_var_comps(self, Z, groups=None, sigma=None, names=None):
        """Build one or more variance components from the columns of a binary
        grouping matrix and variance specification.
        
        Arguments:
            Z (DataFrame, NDArray): A binary 2D array or pandas DataFrame. Each
                column represents a column/predictor, each row represents an
                observation.
            groups (2DArray): A 2D binary array that maps the columns of Z
                onto variance components. Has dimension n_rows(Z) x k,
                where k is the number of distinct variance components. If None,
                a single group over all columns of Z is assumed.
            sigma (2DArray): A k x k 2D covariance matrix specifying the
                covariances between variance components.
            names (list): Optional list specifying the names of the groups. 
        """
        if groups is None:
            groups = np.ones((Z.shape[1], 1))
        n_grps = groups.shape[1]

        if names is None:
            names = getattr(groups, 'columns',
                            ['VC{}'.format(i) for i in range(n_grps)])

        # Work with array instead of DF
        if hasattr(groups, 'values'):
            groups = groups.values

        for i in range(n_grps):
            z_grp = Z[:, groups[:, i].astype(bool)]
            # TODO: select default prior
            vc = VarComp(names[i], z_grp)
            self.add_term(vc)

    def add_term(self, term):
        if term.name in self.terms:
            raise ValueError("Term with name {} already exists!"
                             .format(term.name))
        self.terms[term.name] = term

    @property
    def X(self):
        """Return X design matrix (i.e., fixed component of model)."""
        pass

    @property
    def Z(self):
        """Return Z design matrix (i.e., random effects/variance components).
        """
        pass

    @property
    def fixed_terms(self):
        return [t for t in self.terms.values() if not isinstance(t, VarComp)]

    @property
    def random_terms(self):
        return [t for t in self.terms.values() if isinstance(t, VarComp)]

    @classmethod
    def from_collection(cls, collection, model):
        """ Initialize a GLMMSpec instance from a BIDSVariableCollection and
        a BIDS-StatsModels JSON spec."""

        if isinstance(collection, BIDSRunVariableCollection):
            if not collection.all_dense():
                raise ValueError("Input BIDSRunVariableCollection contains at "
                                 "least one sparse variable. All variables must"
                                 " be dense!")
        glmms = GLMMSpec()

        kwargs = {}

        # Fixed terms
        names = model.get('X', [])
        if names:
            names = collection.match_variables(names)
            X = collection.to_df(names).loc[:, names]
            kwargs['X'] = X

        # Variance components
        vcs = model.get('VarianceComponents', [])
        Z_list = []
        if vcs:
            for vc in vcs:
                # Levels can either be defined by the levels of a single
                # categorical ("LevelsFrom") or by a set of binary variables.
                if 'LevelsFrom' in vc:
                    data = collection.variables[vc['LevelsFrom']].values
                    Z_list.append(pd.get_dummies(data).values)
                else:
                    names = collection.match_variables(vc['Levels'])
                    df = collection.to_df(names).loc[:, names]
                    Z_list.append(df.values)

            Z = np.concatenate(Z_list, axis=1)
            groups = np.zeros((Z.shape[1], len(Z_list)))
            c = 0
            for i, vc in enumerate(Z_list):
                n = vc.shape[1]
                groups[c:(c+n), i] = 1
                c += n
            groups = pd.DataFrame(groups, columns=[vc['Name'] for vc in vcs])

            kwargs['Z'] = Z
            kwargs['groups'] = groups
        
        error = model.get('Error')
        if error:
            kwargs['family'] = error.get('Family')
            kwargs['link'] = error.get('Link')

        return GLMMSpec(**kwargs)


class Term(object):

    def __init__(self, name, values, categorical=False, prior=None,
                 metadata=None):
        self.name = name
        self.values = values
        self.categorical = categorical
        self.prior = prior
        self.metadata = metadata or {}


class VarComp(Term):

    def __init__(self, name, values, prior=None, metadata=None):
        super(VarComp, self).__init__(name, values, categorical=True,
                                      prior=prior, metadata=metadata)
        self.index_vec = self.dummies_to_vec(values)

    @staticmethod
    def dummies_to_vec(dummies):
        """
        For the sake of computational efficiency (i.e., to avoid lots of
        large matrix multiplications in the backends), invert the dummy-coding
        process and represent full-rank dummies as a vector of indices into the
        coefficients.
        """
        vec = np.zeros(len(dummies), dtype=int)
        for i in range(dummies.shape[1]):
            vec[(dummies[:, i] == 1)] = i + 1
        return vec


class Prior(object):
    pass
