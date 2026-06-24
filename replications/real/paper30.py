## This file contains the replication of paper 30
from pathlib import Path
import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 30

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper30(title, dataset_name, query_li, id_li, debug=False, use_pyfixest=True):
    """
    Replicates the analysis of paper 30
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "late_start"
    outcome_var = "Zmarks"
    time_var = "year"
    control_vars = ["num_month", "order_ass"]
    method = "ols"

    df["course"] = df["grade"].astype(str) + " " + df["subject"].astype(str) + " " + df["cohort"].astype(str)
    df["class_fe"] = df["year"].astype(str) + " " + df["course"]
    df["id"] = df["id"].astype(str)

    data = df[[outcome_var, treat_var, "id", "class_fe"] + control_vars].dropna()

    # The standard smf.ols solution with many fixed effects takes very long; use pyfixest instead
    if use_pyfixest:
        model = pf.feols("Zmarks ~ i(late_start) + num_month + order_ass | id + class_fe",
            data=data, vcov={"CRV1": "id + class_fe"})
        answer = model.coef()["late_start::1"]
        std_err = model.se()["late_start::1"]
    else:
        smf_formula = f"{outcome_var} ~ C({treat_var}) + C(id) + {' + '.join(control_vars)} + C(class_fe)"
        model = smf.ols(smf_formula, data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data[["id", "class_fe"]]})
        answer = model.params["C(late_start)[T.1]"]
        std_err = model.bse["C(late_start)[T.1]"]

    if debug:
        print(f"answer={answer:.4f}, se={std_err:.4f}")

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, time_var=time_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper30(debug=False):
    """
    Builds the representation of paper 30

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Double-shift schooling and student success: Quasi-experimental evidence from Europe"
    dataset_name = "lusher_double_shift"
    year = 2016
    domain = "economics"
    n_solutions = 1

    query1 = "How does being scheduled in the afternoon block affect standardized assignment grade?"

    solutions = replicated_paper30(title, dataset_name, [query1], [51], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
