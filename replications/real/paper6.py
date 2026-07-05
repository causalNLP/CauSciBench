## This file contains the replication of paper 6
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 6

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper6(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 6
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    outcome_var = "demsharenext"
    running_var = "difdemshare"
    treat_var = "right"
    control_vars = ["demofficeexp", "demelectexp", "othofficeexp", "othelectexp"]
    method = "rdd"

    poly_vars = ["difdemshare", "difdemshare2", "difdemshare3", "difdemshare4",
                 "rdifdemshare", "rdifdemshare2", "rdifdemshare3", "rdifdemshare4"]
    cols_needed = [outcome_var, treat_var] + poly_vars + control_vars + ["statedisdec"]
    df = df[cols_needed].dropna()

    formula = (f"demsharenext ~ difdemshare + difdemshare2 + difdemshare3 + difdemshare4 + "
               f"rdifdemshare + rdifdemshare2 + rdifdemshare3 + rdifdemshare4 + "
               f"right + {' + '.join(control_vars)}")
    model = smf.ols(formula=formula, data=df).fit(cov_type="cluster",
                                                  cov_kwds={"groups": df["statedisdec"].astype("int")})
    if debug:
        print(model.summary())
    answer1 = model.params[treat_var]
    std_err1 = model.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, running_var=running_var, is_rct=False)

    solution_dict = {id_li[0]: solution1}

    return solution_dict


def build_paper6(debug=False):
    """
    Builds the representation of paper 6

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Randomized experiments from non-random selection in U.S. House elections"
    dataset_name = "lee_randomized_experiments"
    year = 2008
    domain = "political science"
    is_rct = False
    is_multirct = False
    n_solutions = 1

    query1 = "Does winning an election in a given period influence the party's subsequent electoral success?"

    solutions = replicated_paper6(title, dataset_name, [query1], [10], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, is_multirct,
                  is_rct, n_solutions)
    return paper
