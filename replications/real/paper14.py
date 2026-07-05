## This file contains the replication of paper 14
from pathlib import Path
from linearmodels.iv import IV2SLS
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 14

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper14(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 14
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path)

    df["Pole"] = df["Pole"].astype("category")

    treat_var = "treat"
    instrument_var = "forcing"
    control_vars = ["gender", "birthplace", "age", "religion", "caste", "Pole"]
    method = "iv"

    outcome_vars = ["total_expenditure", "food_expenditure",
                    "education_expenditure", "kerosene_expenditure"]
    keep_vars = control_vars + [treat_var, instrument_var]

    solutions = {}

    for i, outcome_var in enumerate(outcome_vars):
        data_sub = df.dropna(subset=[outcome_var] + keep_vars)
        iv_formula = (f"{outcome_var} ~ 1 + gender + birthplace + age + religion + caste"
                      f" + C(Pole) + [{treat_var} ~ {instrument_var}]")
        model = IV2SLS.from_formula(iv_formula, data=data_sub).fit(
            cov_type="clustered", clusters=data_sub["Pole"])
        if debug:
            print(f"\n--- {outcome_var} ---")
            print(model.summary)
        answer = model.params[treat_var]
        std_err = model.std_errors[treat_var]
        solution = Solution(id_li[i], title, query_li[i], method, dataset_name,
                            answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                            control_vars=control_vars, instrument_var=instrument_var,
                            is_rct=False)
        solutions[id_li[i]] = solution

    return solutions


def build_paper14(debug=False):
    """
    Builds the representation of paper 14

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The effects of rural electrification in india: An instrumental variable approach at the household level"
    dataset_name = "thomas_effects_of"
    year = 2020
    domain = "development economics"
    n_solutions = 4

    query1 = "Does access to electricity lead to an increase in total household expendititure?"
    query2 = "How does access to electric grids affect food expenditures for a household?"
    query3 = "What is the effect of electrification on education expenses?"
    query4 = "Does having access to electricity increase kerosene expenditures?"

    solutions = replicated_paper14(title, dataset_name,
                                   [query1, query2, query3, query4],
                                   [24, 25, 26, 27], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
