## This file contains the replication of paper 96
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 96

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper96(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 96
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv", encoding="latin-1")

    treat_var = "T"
    method = "ols"
    base_controls = ['qualtrics', 'female', 'urban', 'Week', 'block']

    ## Solution 1: effect on Covid-19 knowledge
    d1 = df.dropna(subset=['Knowledge', 'T', 'female', 'urban', 'qualtrics', 'block', 'Week']).copy()
    m1 = smf.ols("Knowledge ~ T + qualtrics + female + urban + C(Week) + C(block)", data=d1).fit(cov_type="cluster", cov_kwds={"groups": d1["block"]})
    if debug:
        print(f"Knowledge: coef={m1.params['T']:.4f}, se={m1.bse['T']:.4f}, N={int(m1.nobs)}")
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         float(m1.params['T']), float(m1.bse['T']),
                         treat_var=treat_var, outcome_var="Knowledge",
                         control_vars=base_controls, is_rct=True)

    ## Solution 2: effect on compliance behavior (T + Long arm + interaction)
    d2 = df.dropna(subset=['Behavior', 'T', 'female', 'urban', 'qualtrics', 'block', 'Week', 'List']).copy()
    d2['TxList'] = d2['T'] * d2['List']
    m2 = smf.ols("Behavior ~ T + List + TxList + qualtrics + female + urban + C(Week) + C(block)", data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2["block"]})
    if debug:
        print(f"Behavior: coef={m2.params['T']:.4f}, se={m2.bse['T']:.4f}, N={int(m2.nobs)}")
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         float(m2.params['T']), float(m2.bse['T']),
                         treat_var=treat_var, outcome_var="Behavior",
                         control_vars=base_controls + ['List', 'TxList'], is_rct=True)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper96(debug=False):
    """
    Builds the representation of paper 96

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Countering misinformation via whatsapp: Preliminary evidence from the covid-19 pandemic in Zimbabwe"
    dataset_name = "bowles_countering"
    year = 2020
    domain = "public health"
    n_solutions = 2

    query1 = "What is the effect of WhatsApp messages on individuals's knowledge about Covid19?"
    query2 = "Does providing credible information improve compliance with Covid guidelines?"

    solutions = replicated_paper96(title, dataset_name, [query1, query2], [174, 175], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
