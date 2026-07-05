## This file contains the replication of paper 70
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 70

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper70(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 70
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df = df.dropna(how="all").copy()

    treat_var = "FAR Best Estimate"
    method = "ols"
    term = 'Q("FAR Best Estimate")'

    def _fit(outcome_var):
        formula = (f'Q("{outcome_var}") ~ Q("FAR Best Estimate") + '
                   'C(Q("Disaster Type")) + C(Continent) + C(Year)')
        m = smf.ols(formula, data=df).fit(cov_type="HC3")
        if debug:
            print(f"{outcome_var}: coef={m.params[term]:.4f}, se={m.bse[term]:.4f}, N={int(m.nobs)}")
        return float(m.params[term]), float(m.bse[term])

    ## Solution 1: effect on total economic damage (US$ Thousands)
    coef1, se1 = _fit("Total Damage (US$ Thousands)")
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         coef1, se1, treat_var=treat_var,
                         outcome_var="Total Damage (US$ Thousands)", is_rct=False)

    ## Solution 2: effect on total deaths
    coef2, se2 = _fit("Total Deaths")
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         coef2, se2, treat_var=treat_var,
                         outcome_var="Total Deaths", is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper70(debug=False):
    """
    Builds the representation of paper 70

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The global costs of extreme weather that are attributable to climate change"
    dataset_name = "newman_global_costs"
    year = 2022
    domain = "environmental science"
    n_solutions = 2

    query1 = "How much of global total economic damages due to extreme weather is attributable to climate change? (in thousands)"
    query2 = "How many global deaths due to extreme weather are attributable to climate change?"

    solutions = replicated_paper70(title, dataset_name, [query1, query2], [116, 117], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
