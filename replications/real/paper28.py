## This file contains the replication of paper 28
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 28

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper28(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 28
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "expansion"
    outcome_var = "ageadjust_drugod"
    state_var = "statefips"
    time_var = "year"
    control_vars = ["unemp_rate", "pov_rate"]
    method = "did"

    data = df[(df[time_var] >= 1999) & (df[time_var] <= 2008)].copy()
    data = data[[outcome_var, treat_var, state_var, time_var, "pop"] + control_vars].dropna().reset_index(drop=True)

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)} + C({state_var}) + C({time_var})"
    cluster = pd.factorize(data[state_var])[0]
    model = smf.wls(formula, data=data, weights=data["pop"]).fit(cov_type="cluster", cov_kwds={"groups": cluster})

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         canonical_did=False, is_rct=False)

    return {id_li[0]: solution1}


def build_paper28(debug=False):
    """
    Builds the representation of paper 28

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Early medicaid expansions and drug overdose mortality in the usa: A quasi-experimental analysis"
    dataset_name = "venkatramani_early_medicaid"
    year = 2018
    domain = "public health"
    n_solutions = 1

    query1 = "What is the effect of Medicaid eligibility expansions on drug overdose mortality per 100,000 among adults aged 25–64 for the years 1999 and 2008?"

    solutions = replicated_paper28(title, dataset_name, [query1], [49], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
