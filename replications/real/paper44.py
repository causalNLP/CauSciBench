## This file contains the replication of paper 44
from pathlib import Path
import pandas as pd
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 44

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper44(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 44
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    outcome_var = "v2x_polyarchy"
    control_vars = ["iwdi_pop_3l", "iwdi_oda_3l", "iwdi_gdppc_3l",
                    "iunhcr_ref_idp_3l", "iwdi_literacy_3l", "iwdi_fuel_3l"]
    method = "iv"

    queries = [("ipema_any_demo_assist_dum_2l", "ipema_any_demo_assdiv_2l"),
        ("iany_demo_all_max_dum_2l", "iany_demo_all_maxdiv_2l")]

    solutions = {}
    for i, (treat_var, instrument_var) in enumerate(queries):
        data = df.copy()
        vars_ = [outcome_var, treat_var, instrument_var] + control_vars
        data = data.dropna(subset=vars_)
        data[vars_] = data.groupby("country")[vars_].transform(lambda x: x - x.mean())

        formula = (f"{outcome_var} ~ -1 + {' + '.join(control_vars)} + "
            f"[{treat_var} ~ {instrument_var}]")
        res = IV2SLS.from_formula(formula, data=data).fit()

        if debug:
            print(res.summary)

        answer = res.params[treat_var]
        std_err = res.std_errors[treat_var]

        solutions[id_li[i]] = Solution(id_li[i], title, query_li[i], method, dataset_name,
                                       answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                                       control_vars=control_vars, instrument_var=instrument_var, is_rct=False)

    return solutions


def build_paper44(debug=False):
    """
    Builds the representation of paper 44

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Computational and robustness reproducibility of \"UN peacekeeping and democratization in conflict-affected countries\""
    dataset_name = "oswald_computational"
    year = 2024
    domain = "political science"
    n_solutions = 2

    query1 = "What is the effect of receiving any (not max) democracy assistance (vs. none) on democracy outcomes?"
    query2 = "What is the effect of receiving the highest level of democracy assistance (vs. less or none) on democracy outcomes?"


    solutions = replicated_paper44(title, dataset_name, [query1, query2], [81, 82], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    
    return paper
