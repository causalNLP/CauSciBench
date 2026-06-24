## This file contains the replication of paper 94
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 94

BASE_DIR = Path("../data/csv_files/realdata/")

SCENARIO_MAP = {'Placebo': 1, 'Treatment 1': 2, 'Treatment 2': 3, 'Treatment 3': 4,
                'Treatment 4': 5, 'Treatment 5': 6, 'Treatment 6': 7}


def _load(dataset_name):
    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df['Scenario_num'] = df['Scenario'].map(SCENARIO_MAP)
    return df


def _fit(df_sub, scenario_num, pref_low, pref_high):
    subset = df_sub[(df_sub['Public_service_preference'] >= pref_low) &
                    (df_sub['Public_service_preference'] < pref_high) &
                    (df_sub['Scenario_num'].isin([1, scenario_num]))].copy()
    subset['treat_indicator'] = (subset['Scenario_num'] == scenario_num).astype(int)
    m = smf.ols("treatment ~ treat_indicator", data=subset).fit(cov_type="HC1")
    return float(m.params["treat_indicator"]), float(m.bse["treat_indicator"])


def replicated_paper94(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 94
    """

    treat_var = "treatment"
    outcome_var = "treatment"
    method = "ols"

    ## Solutions 1-3: nursing home quality (dataset 1)
    df1 = _load("brogaard_interpreting_performance1")

    d1, se1 = _fit(df1, 2, 75, 100)
    if debug:
        print(f"Q1: coef={d1:.4f}, se={se1:.4f}")
    solution1 = Solution(id_li[0], title, query_li[0], method, "brogaard_interpreting_performance1",
                         d1, se1, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    female_df1 = df1[df1['Male_dummy'].str.strip() == 'Female']
    d2, se2 = _fit(female_df1, 2, 75, 100)
    if debug:
        print(f"Q2: coef={d2:.4f}, se={se2:.4f}")
    solution2 = Solution(id_li[1], title, query_li[1], method, "brogaard_interpreting_performance1",
                         d2, se2, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    d3, se3 = _fit(df1, 3, 25, 50)
    if debug:
        print(f"Q3: coef={d3:.4f}, se={se3:.4f}")
    solution3 = Solution(id_li[2], title, query_li[2], method, "brogaard_interpreting_performance1",
                         d3, se3, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    ## Solutions 4-6: refuse collection quality (dataset 2)
    df2 = _load("brogaard_interpreting_performance2")

    d4, se4 = _fit(df1, 4, 75, 100)
    if debug:
        print(f"Q4: coef={d4:.4f}, se={se4:.4f}")
    solution4 = Solution(id_li[3], title, query_li[3], method, "brogaard_interpreting_performance1",
                         d4, se4, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    d5, se5 = _fit(df2, 3, 75, 100)
    if debug:
        print(f"Q5: coef={d5:.4f}, se={se5:.4f}")
    solution5 = Solution(id_li[4], title, query_li[4], method, "brogaard_interpreting_performance2",
                         d5, se5, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    d6, se6 = _fit(df2, 4, 0, 25)
    if debug:
        print(f"Q6: coef={d6:.4f}, se={se6:.4f}")
    solution6 = Solution(id_li[5], title, query_li[5], method, "brogaard_interpreting_performance2",
                         d6, se6, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    return {id_li[0]: solution1, id_li[1]: solution2, id_li[2]: solution3,
            id_li[3]: solution4, id_li[4]: solution5, id_li[5]: solution6}


def build_paper94(debug=False):
    """
    Builds the representation of paper 94

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Interpreting performance information: Motivated reasoning or unbiased comprehension? A replication and extension"
    dataset_name = "brogaard_interpreting_performance1"
    year = 2023
    domain = "psychology"
    n_solutions = 6

    query1 = "What is the effect of provider-type information in the scenario with a better public non-profit vs. worse for-profit on perceived nursing home quality for citizens with public preference between 75 and 100?"
    query2 = "What is the effect of provider-type information assuming a better public non-profit vs. worse for-profit on perceived nursing home quality among female citizens with public preference between 75 and 100?"
    query3 = "How does provider-type information framing for-profit homes as higher quality and public non-profits as lower quality - shape perceptions of nursing home quality among citizens with public service preferences between 25 and 50?"
    query4 = "What is the effect of provider-type information - presenting public non-profits as higher quality and private non-profits as lower quality - on perceived nursing home quality among citizens with public service preferences between 75 and 100?"
    query5 = "What is the effect on perceived refuse collection quality due to provider-type information assuming better for-profit vs. worse public non-profit for citizens with public service preference between 75 and 100?"
    query6 = "What is the effect on perceived refuse collection quality due to provider-type information assuming better public vs. worse for-profit among people with public service preference between 0 and 25?"

    solutions = replicated_paper94(title, dataset_name, [query1, query2, query3, query4, query5, query6], [163, 164, 165, 166, 167, 168],
                                   debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
