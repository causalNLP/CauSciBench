## This file contains the replication of paper 92
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 92

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper92(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 92
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv", encoding="utf-8")
    df = df[(df["treat_deshawn"]==1) | (df["treat_jake"]==1)].copy()

    treat_var = "treat_deshawn"
    outcome_var = "reply_atall"
    method = "ols"

    def _fit(sub):
        m = smf.ols("reply_atall ~ treat_deshawn", data=sub).fit(cov_type="HC1")
        return float(m.params["treat_deshawn"]), float(m.bse["treat_deshawn"])

    ## Solution 1: no partisan signal
    sub1 = df[(df["treat_noprimary"]==1) & (df["treat_demprimary"]==0) & (df["treat_repprimary"]==0)]
    d1, se1 = _fit(sub1)
    if debug:
        print(f"No partisan signal: diff={d1:.4f}, SE={se1:.4f}")

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         d1, se1, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    ## Solution 2: Democratic partisan signal
    d2, se2 = _fit(df[df["treat_demprimary"]==1])
    if debug:
        print(f"Democratic signal: diff={d2:.4f}, SE={se2:.4f}")

    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         d2, se2, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    ## Solution 3: Republican partisan signal
    d3, se3 = _fit(df[df["treat_repprimary"]==1])
    if debug:
        print(f"Republican signal: diff={d3:.4f}, SE={se3:.4f}")

    solution3 = Solution(id_li[2], title, query_li[2], method, dataset_name,
                         d3, se3, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    return {id_li[0]: solution1, id_li[1]: solution2, id_li[2]: solution3}


def build_paper92(debug=False):
    """
    Builds the representation of paper 92

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Do politicians racially discriminate against constituents? A field experiment on state legislators"
    dataset_name = "butler_do_politicians"
    year = 2011
    domain = "political science"
    n_solutions = 3

    query1 = "Do legislators respond differently by race when no partisan signal is included?"
    query2 = "Do legislators respond differently by race when a Democratic partisan signal is included?"
    query3 = "Do legislators respond differently by race when a Republican partisan signal is included?"

    solutions = replicated_paper92(title, dataset_name, [query1, query2, query3], [157, 158, 159], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
