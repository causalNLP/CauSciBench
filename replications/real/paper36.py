## This file contains the replication of paper 36
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 36

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper36(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 36
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "sib"
    outcome_var = "year_edu"
    control_vars = ["han", "gender", "urban_father", "edu_father", "job_father", "job_mother",
                    "party_father", "party_mother"]
    method = "ols"

    edu_order = {"没有上过学": 0, "小学": 1, "初中": 2, "高中或职高": 3, "专科": 4, "本科": 5, "研究生": 6}
    df["edu_father"] = df["edu_father"].map(edu_order)
    df["party_father"] = df["party_father"].map({"群众": 0, "党员": 1})
    df["party_mother"] = df["party_mother"].map({"群众": 0, "党员": 1})

    data = df[[outcome_var, treat_var] + control_vars].dropna()

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)}"
    model = smf.ols(formula, data=data).fit()

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper36(debug=False):
    """
    Builds the representation of paper 36

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The effect of number of siblings and birth order on educational attainment: Empirical evidence from Chinese general social survey"
    dataset_name = "xiong_effect_of"
    year = 2020
    domain = "education"
    n_solutions = 1

    query1 = "Does number of sibilings have an effect on the years of education?"

    solutions = replicated_paper36(title, dataset_name, [query1], [57], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
