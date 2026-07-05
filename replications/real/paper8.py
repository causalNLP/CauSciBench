## This file contains the replication of paper 8
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 8

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper8(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 8
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0).reset_index()

    treat_var = "treat_out"
    outcome_var = "responded"
    interaction_var = "leg_black"
    control_vars = ["leg_black", "leg_democrat", "south", "leg_senator", "blackpercent",
                    "black_medianhh", "white_medianhh", "statessquireindex", "urbanpercent",
                    "nonblacknonwhite", "totalpop"]
    method = "ols"

    other_controls = [c for c in control_vars if c != interaction_var]
    formula = (f"{outcome_var} ~ {treat_var} * {interaction_var} + {' + '.join(other_controls)}")

    model = smf.ols(formula, data=df).fit()
    if debug:
        print(model.summary())

    ## Solution 1: effect of treat_out
    answer1 = model.params[treat_var]
    std_err1 = model.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    ## Solution 2: interaction effect of treat_out * leg_black
    interaction_term = f"{treat_var}:{interaction_var}"
    answer2 = model.params[interaction_term]
    std_err2 = model.bse[interaction_term]
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, interaction_var=interaction_var, is_rct=True)

    solution_dict = {id_li[0]: solution1, id_li[1]: solution2}

    return solution_dict


def build_paper8(debug=False):
    """
    Builds the representation of paper 8

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Black politicians are more intrinsically motivated to advance blacks interests: A field experiment manipulating political incentives"
    dataset_name = "broockman_black_politicians"
    year = 2013
    domain = "political science"
    n_solutions = 2

    query1 = "How much differently do Black legislators respond to out of districts email?"
    query2 = "Do legislators tend to respond to out-of-district emails?"

    solutions = replicated_paper8(title, dataset_name, [query1, query2], [13, 14], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
