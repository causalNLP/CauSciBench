## This file contains the replication of paper 48
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 48

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper48(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 48
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df = df[df["donor_code"] == 1]

    treat_var = "str_partnership"
    outcome_var = "total_china_ln"
    control_vars = ["democracy", "taiwan", "agree_us", "chinese_exports_log",
                    "total_deaths_log", "gdp_per_capita_log"]
    method = "ols"

    res = smf.ols(f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)}",
                  data=df).fit(cov_type="HC1")

    if debug:
        print(res.summary())

    answer = res.params[treat_var]
    std_err = res.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper48(debug=False):
    """
    Builds the representation of paper 48

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Chinas foreign aid political drivers: Lessons from a novel dataset of mask diplomacy in latin america during the covid-19 pandemic"
    dataset_name = "urdinez_china"
    year = 2022
    domain = "international relations"
    n_solutions = 1

    query1 = "Does having a strategic partnership with China increase the total Chinese foreign aid a country receives during COVID-19?"

    solutions = replicated_paper48(title, dataset_name, [query1], [87], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
