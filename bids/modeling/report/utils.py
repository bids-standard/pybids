import numpy as np
import pandas as pd
import re
from scipy.linalg import null_space
import warnings

def node_report(node_output):
    """
    Generate a report for a single node.
    """
    # If only intercept model, don't run reports
    if node_output.node.level != 'run':
        return {}

    _report = {
        'VIF': get_all_contrast_vif(node_output),
    }
    try:
        from .viz import plot_design_matrix, plot_corr_matrix
    except ImportError:
        warnings.warn(
            'altair failed to import, and is required for StatsModel report plots.', 
            ImportWarning)
    else:
        _report['design_matrix_plot'] = plot_design_matrix(
            node_output.X, timecourse=True)

        _report['design_matrix_corrplot'] = plot_corr_matrix(node_output.X)

    return _report


def est_vif(desmat):
    '''
    General variance inflation factor estimation.  Calculates VIF for all 
    regressors in the design matrix.

    Parameters
    ----------
        desmat (DataFrame): design matrix. Intercept not required.

    Returns
    -------
        vif_data (DataFrame): Variance inflation factor for each regressor in the design matrix
                generally goal is VIF<5
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    desmat_with_intercept = desmat.copy()
    desmat_with_intercept['intercept'] = 1
    vif_data = pd.DataFrame()
    vif_data['regressor'] = desmat_with_intercept.columns.drop('intercept')
    vif_data['VIF'] = [variance_inflation_factor(desmat_with_intercept.values, i)
                          for i in range(len(desmat_with_intercept.columns))
                          if desmat_with_intercept.columns[i] != 'intercept']
    return vif_data


def get_eff_reg_vif(desmat, contrast_def):
    '''
    The goal of this function is to estimate a variance inflation factor for a contrast.
    This is done by extending the effective regressor definition from Smith et al (2007)
    Meaningful design and contrast estimability (NeuroImage).  Regressors involved
    in the contrast estimate are rotated to span the same space as the original space
    consisting of the effective regressor and and an orthogonal basis.  The rest of the 
    regressors are unchanged.

    Parameters
    ----------
        desmat (DataFrame): design matrix.  Assumed to be a pandas dataframe with column  
             headings which are used define the contrast of interest
        contrast_def (array or Series): a single contrast defined as a vector

    Returns
    -------
        vif (float): a single VIF for the contrast of interest  
    '''
    des_nuisance_regs = desmat[desmat.columns[contrast_def == 0]]
    des_contrast_regs = desmat[desmat.columns[contrast_def != 0]]

    con = np.atleast_2d(contrast_def[contrast_def != 0])
    con2_t = null_space(con)
    con_t = np.transpose(con)
    x = des_contrast_regs.copy().values
    q = np.linalg.pinv(np.transpose(x)@ x)
    f1 = np.linalg.pinv(con @ q @ con_t)
    pc = con_t @ f1 @ con @ q
    con3_t = con2_t - pc @ con2_t
    f3 = np.linalg.pinv(np.transpose(con3_t) @ q @ con3_t)
    eff_reg = x @ q @ np.transpose(con) @ f1
    eff_reg = pd.DataFrame(eff_reg, columns = [0])

    other_reg = x @ q @ con3_t @ f3 
    other_reg_names = [f'orth_proj{val}' for val in range(other_reg.shape[1])]
    other_reg = pd.DataFrame(other_reg, columns = other_reg_names)

    des_for_vif = pd.concat([eff_reg, other_reg, des_nuisance_regs], axis = 1)
    vif_dat = est_vif(des_for_vif)
    vif_output = vif_dat[vif_dat.regressor == 0].VIF.values[0]
    return vif_output


def generate_contrast_matrix(contrasts, cols):
    """Generate a contrast matrix from a list of contrast definitions.

    Parameters
    ----------
    contrasts : list of ContrastInfo objects
    cols : list of design matrix columns

    Returns
    -------
    contrast_matrix : 2D array

    """

    rows = []
    ix = []
    for con in contrasts:
        vec = cols.map(dict(zip(con.conditions, con.weights))).fillna(0)
        rows.append(vec.tolist())
        ix.append(con.name)

    df = pd.DataFrame(rows, columns=cols.tolist(), index=ix)
    return df


def get_all_contrast_vif(node_output):
    '''
    Calculates the VIF for multiple contrasts

    Parameters
    ----------
        node_output (BIDSStatsModelNodeOutput): Node output to compute VIF for.

    Returns
    -------
        vif_contrasts (DataFrame): Data frame containing the VIFs for all contrasts.
    '''
    vif_contrasts = {'contrast': [],
                      'VIF': []}
    
    con_matrix = generate_contrast_matrix(
        node_output.contrasts, node_output.X.columns)
    for name, weights in con_matrix.iterrows():
        # Transform weights to vector matching X's columns
        vif_out = get_eff_reg_vif(node_output.X, weights)
        vif_contrasts['contrast'].append(name)
        vif_contrasts['VIF'].append(vif_out) 
    vif_contrasts = pd.DataFrame(vif_contrasts)
    return vif_contrasts     


def deroot(val, root):
    if isinstance(val, str):
        if val.startswith(root):
            idx = len(root)
            if val[idx] == '/':
                idx += 1
            val = val[idx:]
    elif isinstance(val, list):
        val = [deroot(elem, root) for elem in val]
    elif isinstance(val, dict):
        val = {key: deroot(value, root) for key, value in val.items()}

    return val

def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def displayify(contrast_name):
    for match, repl in (('_gt_', ' &gt; '), ('_lt_', ' &lt; '), ('_vs_', ' vs. ')):
        contrast_name = contrast_name.replace(match, repl)
    return contrast_name

def to_alphanum(string):
    """Convert string to alphanumeric

    Replaces all other characters with underscores and then converts to camelCase

    Examples
    --------

    >>> to_alphanum('abc123')
    'abc123'
    >>> to_alphanum('a four word phrase')
    'aFourWordPhrase'
    >>> to_alphanum('hyphen-separated')
    'hyphenSeparated'
    >>> to_alphanum('object.attribute')
    'objectAttribute'
    >>> to_alphanum('array[index]')
    'arrayIndex'
    >>> to_alphanum('array[0]')
    'array0'
    >>> to_alphanum('snake_case')
    'snakeCase'
    """
    return snake_to_camel(re.sub("[^a-zA-Z0-9]", "_", string))