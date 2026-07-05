## This file contains the replication of paper 24
from pathlib import Path
import statsmodels.api as sm
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 24

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper24(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 24
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "dpropertynz"
    outcome_var = "lzri_sfr"
    state_var = "regionid"
    time_var = "mdate"
    method = "did"

    cond = (df["propertynz"] == 1) & (df["window"] >= -12) & (df["window"] <= 12)
    did_data = df.loc[cond, [outcome_var, "post", treat_var, state_var, "county", time_var]].dropna()
    did_data = did_data.copy()
    did_data["county_mdate"] = did_data["county"].astype(str) + "#" + did_data[time_var].astype(str)

    demean_cols = [outcome_var, "post", treat_var]
    tmp = did_data.copy()
    tmp[[c + "_dm1" for c in demean_cols]] = tmp.groupby("county_mdate")[demean_cols].transform(lambda g: g - g.mean())
    tmp[[c + "_dm2" for c in demean_cols]] = tmp.groupby(state_var)[[c + "_dm1" for c in demean_cols]].transform(lambda g: g - g.mean())

    y = tmp[f"{outcome_var}_dm2"]
    X = tmp[[f"post_dm2", f"{treat_var}_dm2"]].copy()
    X[f"post:{treat_var}_dm2"] = tmp["post_dm2"] * tmp[f"{treat_var}_dm2"]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": did_data[state_var]})

    if debug:
        print(model.summary())

    did_coef = f"post:{treat_var}_dm2"
    answer = model.params[did_coef]
    std_err = model.bse[did_coef]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var, canonical_did=False, is_rct=False)

    return {id_li[0]: solution1}


def build_paper24(debug=False):
    """
    Builds the representation of paper 24

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Do wall street landlords undermine renters welfare?"
    dataset_name = "gurun_do_wall_street"
    year = 2022
    domain = "finance"
    n_solutions = 1

    query1 = "What is the effect of institutional landlord mergers on neighborhood rents?"

    solutions = replicated_paper24(title, dataset_name, [query1], [45], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
