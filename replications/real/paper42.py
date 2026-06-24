## This file contains the replication of paper 42
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 42

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper42(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 42
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "LD_ln_cumfdigdp"
    outcome_var = "D_EBRD"
    instrument_var = "L2_change_predX2a"
    control_vars = ["D_postcomseats", "D_polcon3", "wartorn", "D_eu", "D_ln_gdpcap"]
    method = "iv"

    df = df.sort_values(["country_", "year"]).copy()

    df["D_EBRD"] = df.groupby("country_")["EBRD"].diff()
    df["D_postcomseats"] = df.groupby("country_")["postcomseats"].diff()
    df["D_polcon3"] = df.groupby("country_")["polcon3"].diff()
    df["D_eu"] = df.groupby("country_")["eu"].diff()
    df["D_ln_gdpcap"] = df.groupby("country_")["ln_gdpcap"].diff()

    df["D_ln_cumfdigdp"] = df.groupby("country_")["ln_cumfdigdp"].diff()
    df["LD_ln_cumfdigdp"] = df.groupby("country_")["D_ln_cumfdigdp"].shift(1)
    df["L2_change_predX2a"] = df.groupby("country_")["change_predX2a"].shift(2)

    time_fe = [c for c in df.columns if c.startswith("_Itime_")]
    country_fe = [c for c in df.columns if c.startswith("_Icountry__")]

    vars_to_use = [outcome_var, treat_var, instrument_var] + control_vars + time_fe + country_fe
    reg = df[vars_to_use].dropna().copy()

    y = reg[outcome_var]
    exog = sm.add_constant(reg[control_vars + time_fe + country_fe])
    endog = reg[treat_var]
    instr = reg[[instrument_var]]

    res = IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instr).fit(cov_type="robust")

    if debug:
        print(res.summary)

    answer = res.params[treat_var]
    std_err = res.std_errors[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper42(debug=False):
    """
    Builds the representation of paper 42

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Foreign direct investors as agents of economic transition: An instrumental variables analysis"
    dataset_name = "malesky_foreign"
    year = 2009
    domain = "political economics"
    n_solutions = 1

    query1 = "How do foreign direct investment (FDI) inflows influence the progress of institutional reforms in transition economies?"

    solutions = replicated_paper42(title, dataset_name, [query1], [71], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
