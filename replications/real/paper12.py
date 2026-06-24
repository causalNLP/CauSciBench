## This file contains the replication of paper 12
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 12

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper12(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 12
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0).reset_index()

    treat_var = "any"
    outcome_var = "got"
    control_vars = ["male", "hiv2004", "age", "rumphi", "balaka"]
    method = "ols"

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)}"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["villnum"]})
    if debug:
        print(model.summary())

    answer1 = model.params[treat_var]
    std_err1 = model.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    return {id_li[0]: solution1}


def build_paper12(debug=False):
    """
    Builds the representation of paper 12

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The Demand for, and Impact of, Learning HIV Status"
    dataset_name = "thornton_the_demand"
    year = 2008
    domain = "health economics"
    n_solutions = 1

    query1 = "Does providing monetary incentives lead individuals to obtain their HIV test results?"

    solutions = replicated_paper12(title, dataset_name, [query1], [21], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
