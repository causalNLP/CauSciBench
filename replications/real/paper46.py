## This file contains the replication of paper 46
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 46

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper46(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 46
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "ch"
    outcome_var = "blame"
    method = "ols"

    res = smf.ols("blame ~ ch", data=df).fit()

    if debug:
        print(res.summary())

    answer = res.params[treat_var]
    std_err = res.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         is_rct=True)

    return {id_li[0]: solution1}


def build_paper46(debug=False):
    """
    Builds the representation of paper 46

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Does mislabeling covid-19 elicit the perception of threat and reduce blame?"
    dataset_name = "xu_does_mislabelling"
    year = 2021
    domain = "public health"
    n_solutions = 1

    query1 = "Does labeling COVID-19 as the 'Chinese Virus' causally affect the level of blame the public assigns to the federal government?"

    solutions = replicated_paper46(title, dataset_name, [query1], [84], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
