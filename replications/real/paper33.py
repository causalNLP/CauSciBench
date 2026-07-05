## This file contains the replication of paper 33
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from solution import Solution, Paper

PAPER_ID = 33

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper33(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 33
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "over"
    outcome_var = "visit_all_np_r"
    running_var = "months_21"
    control_vars = ["dummy21", "age_c", "age_c_sq", "age_c_post", "age_c_post_sq"]
    method = "rdd"

    df["pop_all_np"] = 1 + 0.00104 * (df["months_21"] / 12)

    data = df[(df["months_21"] >= -24) & (df["months_21"] <= 23)].copy()

    y = data[outcome_var]
    X = sm.add_constant(data[[treat_var] + control_vars])
    w = data["pop_all_np"]

    model = sm.WLS(y, X, weights=w).fit(cov_type="HC1")

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper33(debug=False):
    """
    Builds the representation of paper 33

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The minimum legal drinking age and morbidity in the US"
    dataset_name = "carpenter_the_minimum_legal"
    year = 2017
    domain = "health economics"
    n_solutions = 1

    query1 = "Does gaining legal access to alcohol at the U.S. minimum legal drinking age cause an increase in nonfatal morbidity?"

    solutions = replicated_paper33(title, dataset_name, [query1], [54], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
