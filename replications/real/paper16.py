## This file contains the replication of paper 16

from pathlib import Path
import statsmodels.api as sm
import pandas as pd
import numpy as np
from solution import Solution, Paper

PAPER_ID = 16

BASE_DIR = Path("../data/csv_files/realdata/")


def propensity_match(df, treatment, outcome, confounders, exact_vars, matches=1):
    """
    Propensity-score matching estimator returning the ATT and its standard error.

    Args:
        df (pd.DataFrame): The dataset
        treatment (str): Name of the treatment variable
        outcome (str): Name of the outcome variable
        confounders (List[str]): Names of the confounding variables used to estimate propensity scores
        exact_vars (List[str]): Names of the variables whose values must agree between treated and control units
        matches (int): Number of control units to match to each treated unit

    Returns:
        (float, float): ATT and its standard error
    """

    X = df[confounders].values
    T = df[treatment].values
    Y = df[outcome].values

    ps = sm.Logit(T, sm.add_constant(X)).fit(disp=0).predict()

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    # Filtration: drop units outside common propensity score support
    ps_min = ps[treated_idx].min()
    ps_max = ps[treated_idx].max()
    control_idx = control_idx[ (ps[control_idx] >= ps_min) & (ps[control_idx] <= ps_max)]
    treated_idx = treated_idx[(ps[treated_idx] >= ps[control_idx].min()) & (ps[treated_idx] <= ps[control_idx].max())]

    individual_effects = []
    used_control = set()

    for i in treated_idx:
        candidates = control_idx[~np.isin(control_idx, list(used_control))]
        for v in exact_vars:
            col = df[v].values
            candidates = candidates[col[candidates] == col[i]]

        if len(candidates) == 0:
            continue

        nearest = np.argsort(np.abs(ps[candidates] - ps[i]))[:matches]
        chosen = candidates[nearest]

        individual_effects.append(Y[i] - Y[chosen].mean())
        used_control.update(chosen.tolist())

    att = np.mean(individual_effects)
    std_error = np.std(individual_effects, ddof=1) / np.sqrt(len(individual_effects))

    return att, std_error


def replicated_paper16(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 16
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path)

    treat_var = "demsnmaj"
    outcome_var = "acttime"
    method = "matching"

    exact_vars = ["lethal", "acutediz", "hosp01", "femdiz01", "mandiz01", "peddiz01"]
    control_vars = ["hospdisc", "natreg", "stafcder", "prevgenx", "hhosleng",
                    "condavg3", "orderent", "vandavg3", "wpnoavg3", "lethal",
                    "deathrt1", "hosp01", "femdiz01", "mandiz01", "peddiz01",
                    "acutediz", "orphdum"]

    att, std_err = propensity_match(df, treat_var, outcome_var, control_vars,
                                     exact_vars=exact_vars)
    if debug:
        print(f"ATT: {att:.4f}, SE: {std_err:.4f}")

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         att, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper16(debug=False):
    """
    Builds the representation of paper 16

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference"
    dataset_name = "ho_matching"
    year = 2007
    domain = "political science"
    n_solutions = 1

    query1 = "Does having a Democratic senate majority speed up the approval times for new drugs?"

    solutions = replicated_paper16(title, dataset_name, [query1], [29], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
