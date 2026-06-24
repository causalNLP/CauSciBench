## This file contains the replication of paper 21
from pathlib import Path
import pandas as pd
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 21

BASE_DIR = Path("../data/csv_files/realdata/")

def replicated_paper21(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 21
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "v4"
    outcome_var = "v9"
    instrument_var = "v18"
    control_vars = ["v19", "v20", "v10", "v11", "v12", "v13", "v21", "v24", "v25"]
    method = "iv"

    df_sample = df[(df["v27"] >= 30) & (df["v27"] <= 39)].copy()

    for q in [1, 2, 3]:
        df_sample[f"Q{q}"] = (df_sample["v18"] == q).astype(int)

    yob_dummies = pd.get_dummies(df_sample["v27"], prefix="YOB").astype(float)
    yob_cols = [c for c in yob_dummies.columns if c != "YOB_39"]
    df_sample = pd.concat([df_sample, yob_dummies], axis=1)

    instr_cols = []
    for q in [1, 2, 3]:
        for year in range(30, 40):
            col = f"Q{q}_Y{year}"
            df_sample[col] = df_sample[f"Q{q}"] * (df_sample["v27"] == year).astype(int)
            instr_cols.append(col)

    df_sample["const"] = 1.0

    model = IV2SLS(dependent=df_sample[outcome_var], exog=df_sample[["const"] + yob_cols + control_vars],
                   endog=df_sample[[treat_var]], instruments=df_sample[instr_cols]).fit(cov_type="unadjusted")

    if debug:
        print(model.summary)

    answer = model.params[treat_var]
    std_err = model.std_errors[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper21(debug=False):
    """
    Builds the representation of paper 21

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Does Compulsory School Attendance Affect Schooling and Earnings?"
    dataset_name = "angrist_does_compulsory"
    year = 1991
    domain = "labor economics"
    n_solutions = 1

    query1 = "What is the effect of an additional year of schooling on weekly earnings for men born in 1930s?"

    solutions = replicated_paper21(title, dataset_name, [query1], [41], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
