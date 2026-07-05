## This file contains the replication of paper 20
from pathlib import Path
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 20

BASE_DIR = Path("../data/csv_files/realdata/")


def replicate_analysis(title, dataset_name, query_li, id_li,
                     outcome_vars, treat_var, instrument_var, df, ols_index=0, debug=False):
    """
    Replicates the analysis of the paper. The statistical approach is the same for the
    two datasets that have been analyzed. Hence, we define a common function to handle this.

    Args:
        title (str): The title of the paper
        dataset_name (str): The name of the dataset
        query_li (List[str]): List of queries
        ref_li (List[str]): List of references
        id_li (List[int]): List of IDs
        outcome_vars ([List[str]]): List of outcome variables, one for each query
        treat_var (str): The treatment variable
        instrument_var (str): The instrument variable
        df (pd.DataFrame): The input dataset
        ols_index (int): Index in outcome_vars corresponding to the OLS (reduced-form) model
        debug (bool, optional): Whether to print full model summaries; used for debugging.
                                Defaults to False.
    """
    solutions = {}

    ols_outcome = outcome_vars[ols_index]
    model_ols = smf.ols(f"{ols_outcome} ~ {instrument_var}", data=df).fit()
    if debug:
        print(model_ols.summary())
    solutions[id_li[ols_index]] = Solution(id_li[ols_index], title, query_li[ols_index], "ols", dataset_name,
                                           model_ols.params[instrument_var], model_ols.bse[instrument_var],
                                           treat_var=instrument_var, outcome_var=ols_outcome, is_rct=True)

    for i in range(len(outcome_vars)):
        if i == ols_index:
            continue
        data = df[[outcome_vars[i], treat_var, instrument_var]].dropna()
        model = IV2SLS.from_formula(
            f"{outcome_vars[i]} ~ 1 + [{treat_var} ~ {instrument_var}]", data=data).fit()
        if debug:
            print(model.summary)
        solutions[id_li[i]] = Solution(id_li[i], title, query_li[i], "iv", dataset_name,
                                       model.params[treat_var], model.std_errors[treat_var],
                                       treat_var=treat_var, outcome_var=outcome_vars[i],
                                       instrument_var=instrument_var, is_rct=True)

    return solutions


def replicated_paper20(title, dataset_name1, dataset_name2,
                       query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 20 (two studies)
    """

    instrument_var = "trustprime"
    treat_var = "trust"

    df1 = pd.read_csv(BASE_DIR / f"{dataset_name1}.csv")
    solutions1 = replicate_analysis(title, dataset_name1, query_li[:4], id_li[:4],
                                    outcome_vars=["private", "community", "coproduce", "trust"],
                                    treat_var=treat_var, instrument_var=instrument_var,
                                    df=df1, ols_index=3, debug=debug)

    df2 = pd.read_csv(BASE_DIR / f"{dataset_name2}.csv")
    solutions2 = replicate_analysis(title, dataset_name2, query_li[4:], id_li[4:],
                                    outcome_vars=["trust", "PPP", "PCP", "coproduce"],
                                    treat_var=treat_var, instrument_var=instrument_var,
                                    df=df2, ols_index=0, debug=debug)

    return {**solutions1, **solutions2}


def build_paper20(debug=False):
    """
    Builds the representation of paper 20

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Public trust and collaborative governance: An instrumental variable approach"
    dataset_name1 = "liu_public_trust1"
    dataset_name2 = "liu_public_trust2"
    year = 2024
    domain = "government"
    n_solutions = 8

    query1 = "Does trust in government cause increased support for public-private partnerships?"
    query2 = "Does trust in government cause increased support for public-citizen partnership?"
    query3 = "Does trust in government cause an increased willingness to coproduce?"
    query4 = "Does change to a different government integrity information affect trust in government?"

    query5 = "Does exposure to government integrity information (corrupt, control, honest) causally increase trust in local government?"
    query6 = "Does an increase in trust in local government increase support for public-private partnerships?"
    query7 = "Does an increase in trust in local government increase support for public-citizen partnerships?"
    query8 = "Does an increase in trust in local government increase citizens' willingness to co-produce policy with local government?"


    solutions = replicated_paper20(title, dataset_name1, dataset_name2,
        [query1, query2, query3, query4, query5, query6, query7, query8],
        [33, 34, 35, 36, 37, 38, 39, 40], debug=debug)

    paper = Paper(PAPER_ID, title, dataset_name1, year, domain, solutions,
                  n_solutions=n_solutions)
    return paper
