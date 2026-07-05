## This file contains the replication of paper 27
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 27

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper27(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 27
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "dwin"
    outcome_var = "dv_p_t2"
    running_var = "dv_c_t1"
    method = "rdd"
    band = 0.25

    drop = ((df["t2_3rdpartyinc"] == 1) | (df["t2_incumbent_has_switched_prty"] == 1) |
        (df["t2_specialelectiontoeelect"] == 1) | (df["t1_atlargeormulticandidate"] == 1) |
        (df["t2_redist"] == 1))
    data = df.loc[~drop].copy()

    data["dwin"] = (data[running_var] >= 0).astype(int)
    for p in range(2, 5):
        data[f"dv_c_t1_{p}"] = data[running_var] ** p
    data["i_dv_c_t1"] = data[running_var] * data["dwin"]
    for p in range(2, 5):
        data[f"i_dv_c_t1_{p}"] = data[f"dv_c_t1_{p}"] * data["dwin"]
    data["margin"] = data[running_var].abs()

    data2 = data.loc[(data["t2_is_midterm"] != 1) & (data["t2_year"] != 2008)].copy()
    sample = data2.loc[data2["margin"] < band]

    formula = (f"{outcome_var} ~ {treat_var}"
               f" + dv_c_t1 + dv_c_t1_2 + dv_c_t1_3 + dv_c_t1_4"
               f" + i_dv_c_t1 + i_dv_c_t1_2 + i_dv_c_t1_3 + i_dv_c_t1_4")
    model = smf.ols(formula, data=sample).fit(cov_type="HC1")

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper27(debug=False):
    """
    Builds the representation of paper 27

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Do congressional candidates have reverse coattails? Evidence from a regression discontinuity design"
    dataset_name = "broockman_do_congressional"
    year = 2009
    domain = "government"
    n_solutions = 1

    query1 = "What is the effect of having a congressional incumbent on the party's presidential vote share in that district at the next election?"

    solutions = replicated_paper27(title, dataset_name, [query1], [48], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
