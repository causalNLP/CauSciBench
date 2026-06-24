## This file contains the replication of paper 90
from pathlib import Path
import numpy as np
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 90

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper90(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 90
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    price = df['treatment_price_normalized'].values
    y = df['outcome_demand'].values
    z = df['instrument_z'].values
    x_cols = [c for c in df.columns if c.startswith('covariate_emotion_')] + ['temporal_time']
    x = df[x_cols].values

    treat_var = 'treatment_price_normalized'
    outcome_var = 'outcome_demand'
    instrument_var = 'instrument_z'
    control_vars = x_cols
    method = 'iv'

    ## Stage 1: price ~ z + x
    X1 = np.column_stack([np.ones(len(price)), z, x])
    price_hat = X1 @ np.linalg.lstsq(X1, price, rcond=None)[0]

    ## Stage 2: y ~ price_hat + x
    X2 = np.column_stack([np.ones(len(y)), price_hat, x])
    beta2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    resid = y - X2 @ beta2
    se2 = np.sqrt(np.diag(np.mean(resid**2) * np.linalg.inv(X2.T @ X2)))

    price_coef = float(beta2[1])
    price_se = float(se2[1])

    if debug:
        p25, p75 = np.percentile(price, 25), np.percentile(price, 75)
        ate = price_coef * (p75 - p25)
        print(f"price_coef={price_coef:.4f}, se={price_se:.4f}, ATE={ate:.4f}")

    ## Solution 1: 2SLS effect of price on demand
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         price_coef, price_se,
                         treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper90(debug=False):
    """
    Builds the representation of paper 90

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Deep IV: A Flexible Approach for Counterfactual Prediction"
    dataset_name = "hartford_deepiv"
    year = 2017
    domain = "economics"
    n_solutions = 1

    query1 = "How does increase in product cost affect customer demand?"

    solutions = replicated_paper90(title, dataset_name, [query1], [153], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
