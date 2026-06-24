## This file contains the replication of paper 88
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from solution import Solution, Paper

PAPER_ID = 88

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper88(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 88
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    d = df[(df["year"] >= 1894) & (df["year"] <= 1910) & (~df["year"].isin([1899, 1905]))].copy()

    yvar = "F1to5billbudgetdummy"
    treat = "budget"
    instr = "bureauotherbudgetincumbent"
    controls = ["age", "age2", "inscrits", "inscrits2", "permargin", "permargin2",
                "cummyears", "cummyears2", "proprietaire", "lib_all", "civil",
                "paris", "budgetincumbent"]

    num_cols = [yvar, treat, instr, "year"] + controls
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=num_cols).copy()

    year_dummies = pd.get_dummies(d["year"].astype(int), prefix="year", drop_first=True)
    d = d.join(year_dummies)

    X1 = sm.add_constant(d[[instr] + controls].join(year_dummies).astype(float))
    m1 = sm.OLS(d[treat].astype(float), X1).fit()
    d["budget_hat"] = m1.fittedvalues

    X2 = sm.add_constant(d[["budget_hat"] + controls].join(year_dummies).astype(float))
    m2 = sm.OLS(d[yvar].astype(float), X2).fit(cov_type="HC1")

    if debug:
        print(m2.summary())

    ## Solution 1: IV effect of budget committee service on future cabinet appointment
    solution1 = Solution(id_li[0], title, query_li[0], "iv", dataset_name,
                         float(m2.params["budget_hat"]), float(m2.bse["budget_hat"]),
                         treat_var=treat, outcome_var=yvar,
                         control_vars=controls, instrument_var=instr, is_rct=False)

    return {id_li[0]: solution1}


def build_paper88(debug=False):
    """
    Builds the representation of paper 88

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Cabinets, committees, and careers: The causal effect of committee service"
    dataset_name = "cirone_cabinets"
    year = 2018
    domain = "political science"
    n_solutions = 1

    query1 = "What is the effect of being appointed to the budget committee on the probability a deputy sponsors a budget amendment within the next five years?"

    solutions = replicated_paper88(title, dataset_name, [query1], [151], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
