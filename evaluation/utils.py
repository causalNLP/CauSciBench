## This file contains utility functions for evaluation

import numpy as np


def standardize_method_name(method):
    """
    Standardize the method names to a common format. The standardization is based on the outputs from LLMs.
    We strongly recommend manual inspection of the outputs, and update this function to account for method names not described here.
    Likewise, please check the mappings for inconsistencies. Since the design is based on heuristics, it may not be perfect.

    Args:
        method (str): The method name to standardize

    Returns:
        (str): The standardized method name
    """

    if method is None:
        return np.nan
    if type(method) != str:
        return np.nan
    method = method.lower()

    if "weighting" in method or 'ipw' in method or 'propensity' in method:
        return "ps"
    elif "front" in method or 'frontdoor' in method:
        return "fd"
    elif "discontinuity" in method or 'fuzzy' in method or 'rdd' in method:
        return "rdd"
    elif "in-difference" in method or "did" in method or "in-diff" in method or 'fixed effects' in method or 'panel' in method:
        return "did"
    elif "matching" in method or "observational" in method:
        return "ps"
    elif "logistic" in method or 'probit' in method or 'logit' in method or 'glm' in method:
        return "glm"
    elif "linear" in method or "means" in method or 'ordinary' in method or 'rct' in method or 'ols' in method or 'wls' in method or 'mean' in method:
        return "ols"
    elif "instrument" in method or "encouragement" in method or "2sls" in method or "iv" in method:
        return "iv"
    elif 'null' in method or 'na' in method or 'n/a' in method or 'none' in method:
        return np.nan
    else:
        return 'other'
