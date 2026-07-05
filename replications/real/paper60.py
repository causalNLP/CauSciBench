## This file contains the replication of paper 60
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 60

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper60(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 60
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    subset_analysis = df[df['Contested_2006'] == 1].copy()

    treat_var = 'Grp_Buy'
    outcome_var = 'Votesharechange'
    control_vars = ['Strata70', 'Strata90', 'Partisan', 'To_Prev', 'Statewide_2005']
    method = 'ols'

    ## Solution 1: effect of radio ads on incumbent vote share change (2006)
    formula1 = 'Votesharechange ~ Grp_Buy + Strata70 + Strata90 + Partisan + To_Prev + Statewide_2005'
    model1 = smf.ols(formula1, data=subset_analysis[subset_analysis['Year_Exp'] == 2006]).fit()
    if debug:
        print(model1.summary())
    answer1 = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    return {id_li[0]: solution1}


def build_paper60(debug=False):
    """
    Builds the representation of paper 60

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Field experiments testing the impact of radio advertisements on electoral competition"
    dataset_name = "panagopolous_field_experiments"
    year = 2008
    domain = "political science"
    is_rct = True
    n_solutions = 1

    query1 = "What was the effect of the radio advertising campaign on the change in the incumbent mayor's vote share in the 2006 elections?"

    solutions = replicated_paper60(title, dataset_name, [query1], [105], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions,
                  is_rct=is_rct, n_solutions=n_solutions)
    
    return paper
