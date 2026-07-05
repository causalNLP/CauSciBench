## This file contains the replication of paper 26
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 26

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper26(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 26
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "gov_dem"
    outcome_var = "unionization"
    running_var = "dem_voteshare"
    state_var = "stfips"
    time_var = "year"
    control_vars = ["leg_dem", "leg_rep", "dem_voteshare", "prscore", "lnpop", "pop15", "pop65", "black"]
    method = "rdd"

    df = df[~df[time_var].isna()].copy()
    df = df[df[time_var] >= 1925]
    df = df[~df[state_var].isna()]
    df = df[df[state_var] <= 56]
    df[time_var] = df[time_var].astype(int)
    df[outcome_var] = df[outcome_var] * 100.0

    sample_mask = ((~df['termyear'].eq(1)) | (df['termyear'].isna())) & df[running_var].between(0.2, 0.8, inclusive='both')
    df_rd = df.loc[sample_mask].copy()

    needed = set([outcome_var, state_var, time_var, treat_var] + control_vars)
    used = df_rd[[c for c in needed if c in df_rd.columns]].dropna()

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)} + C({state_var}) + C({time_var})"
    model = smf.ols(formula, data=used).fit(cov_type="cluster", cov_kwds={"groups": used[state_var]}, use_t=True)

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, running_var=running_var,
                         state_var=state_var, time_var=time_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper26(debug=False):
    """
    Builds the representation of paper 26

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Estimating the impact of gubernatorial partisanship on policy settings and economic outcomes: A regression discontinuity approach"
    dataset_name = "leigh_estimating_the_impact"
    year = 2011
    domain = "political science"
    n_solutions = 1

    query1 = "What is the effect of having a Democratic governor on a state's unionization rate?"

    solutions = replicated_paper26(title, dataset_name, [query1], [47], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
