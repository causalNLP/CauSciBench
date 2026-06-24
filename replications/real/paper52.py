## This file contains the replication of paper 52
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 52

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper52(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 52
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df["ST"] = pd.Categorical(df["state"]).codes + 1
    df = df.sort_values(["ST", "year"]).set_index(["ST", "year"])
    df["L2_policy"] = df.groupby("ST")["policy"].shift(2)
    df = df.reset_index()

    treat_var = "union"
    outcome_var = "lr"
    instrument_var = "RTW"
    method = "iv"

    pct_vars = [col for col in df.columns if col.startswith("pct")]
    analysis_vars = [outcome_var, treat_var, "L2_policy", "employment", "ST", "year", instrument_var] + pct_vars
    df = df[analysis_vars].dropna()

    year_dummies = pd.get_dummies(df["year"], prefix="year", drop_first=True)
    df = pd.concat([df, year_dummies], axis=1)
    year_cols = year_dummies.columns.tolist()

    df = df.dropna()
    state_dummies = pd.get_dummies(df["ST"], prefix="ST", drop_first=True)
    df = pd.concat([df, state_dummies], axis=1)
    state_cols = state_dummies.columns.tolist()

    df_panel = df.set_index(["ST", "year"])
    iv_formula = (f"{outcome_var} ~ 1 + L2_policy + {' + '.join(pct_vars)} + "
        f"{' + '.join(year_cols)} + {' + '.join(state_cols)} + [{treat_var} ~ {instrument_var}]")

    res = IV2SLS.from_formula(iv_formula, df_panel, weights=df_panel["employment"]).fit(
        cov_type="clustered", clusters=df_panel.index.get_level_values("ST"))

    if debug:
        print(res.summary)

    answer = res.params[treat_var]
    std_err = res.std_errors[treat_var]

    control_vars = ["L2_policy"] + pct_vars
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var,
                         state_var="ST", time_var="year", is_rct=False)

    return {id_li[0]: solution1}


def build_paper52(debug=False):
    """
    Builds the representation of paper 52

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Does right to work imperil the right to health? The effect of labour unions on workplace fatalities"
    dataset_name = "zoorob_does_right"
    year = 2018
    domain = "labor economics"
    n_solutions = 1

    query1 = "How does the percentage of unionised workers in a state effect the state's occupational fatality rate per 100,000 workers?"

    solutions = replicated_paper52(title, dataset_name, [query1], [91], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
