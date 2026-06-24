## This file contains the replication of paper 3
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 3

BASE_DIR = Path("../data/csv_files/realdata/")

import numpy as np

def lalonde_replication(df, treatment, outcome, confounders, matches=1):
    """
    Replicates the analysis of the Lalonde dataset using propensity score matching.

    Args:
        df (pd.DataFrame): The dataset
        treatment (str): The name of the treatment variable
        outcome (str): The name of the outcome variable
        confounders (list): A list of the names of the confounding variables
        matches (int): The number of control units to match per treated unit
    Returns:
        (float, float): The ATT and its standard error
    """

    X = df[confounders].values
    T = df[treatment].values
    Y = df[outcome].values
 
    ps = sm.Logit(T, sm.add_constant(X)).fit(disp=0).predict()
 
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
 
    # Filtration: drop control units outside treated propensity score range
    ps_min = ps[treated_idx].min()
    ps_max = ps[treated_idx].max()
    control_idx = control_idx[(ps[control_idx] >= ps_min) & (ps[control_idx] <= ps_max)]
 
    ps_control = ps[control_idx]
 
    individual_effects = []
    for i in treated_idx:
        nearest = np.argsort(np.abs(ps_control - ps[i]))[:matches]
        individual_effects.append(Y[i] - Y[control_idx[nearest]].mean())
 
    att = np.mean(individual_effects)
    std_error = np.std(individual_effects, ddof=1) / np.sqrt(len(individual_effects))

    return att, std_error

def replicated_paper3(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 3
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    treatment_var = "treat"
    outcome_var = "re78"
    method = "matching"

    control_vars = ["age", "education", "black", "hispanic", "nodegree", "re74", "re75", 
                    "age2", "education2", "re742", "re752", "u74", "u75", "u74_black"]
    att, std_err = lalonde_replication(df, treatment_var, outcome_var, control_vars)

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         att, std_err, treat_var=treatment_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    solution_dict = {id_li[0]: solution1}

    return solution_dict


def build_paper3(debug=False):
    """
    Builds the representation of paper 3

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Causal effects in nonexperimental studies"
    dataset_name = "dehejia_causal_effects"
    year = 1999
    domain = "labor economics"
    n_solutions = 1

    query1 = "Does participating in the NSW training program lead to an increase in earnings?"

    solutions = replicated_paper3(title, dataset_name, [query1], [6], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, 
                  n_solutions=n_solutions)
    return paper
