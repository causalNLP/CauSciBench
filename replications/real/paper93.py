## This file contains the replication of paper 93
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 93

BASE_DIR = Path("../data/csv_files/realdata/")

CONTROLS = ["popdens","asian","black","hispanic","other","lesshs","college","fborn",
            "male1524","poverty","unemployed","manufacturing"]


def replicated_paper93(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 93
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "all_cml_r"
    instrument_var = "iv_cml_r"
    method = "iv"

    def _fit(outcome_var):
        need = [treat_var, instrument_var, "place_id", "year", "popweights", outcome_var] + CONTROLS
        d = df.dropna(subset=need).copy()
        f1 = treat_var + " ~ " + instrument_var + " + " + " + ".join(CONTROLS) + " + C(place_id) + C(year)"
        fs = smf.wls(f1, data=d, weights=d["popweights"]).fit()
        d["all_hat"] = fs.fittedvalues
        f2 = outcome_var + " ~ all_hat + " + " + ".join(CONTROLS) + " + C(place_id) + C(year)"
        ss = smf.wls(f2, data=d, weights=d["popweights"]).fit(cov_type="cluster", cov_kwds={"groups": d["place_id"]})
        if debug:
            print(f"{outcome_var}: coef={ss.params['all_hat']:.4f}, se={ss.bse['all_hat']:.4f}, N={int(ss.nobs)}")
        return float(ss.params["all_hat"]), float(ss.bse["all_hat"])

    ## Solution 1: effect on violent crime
    coef1, se1 = _fit("log_viol_r_1ld")
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         coef1, se1, treat_var=treat_var, outcome_var="log_viol_r_1ld",
                         control_vars=CONTROLS, instrument_var=instrument_var, is_rct=False)

    ## Solution 2: effect on homicide rate
    coef2, se2 = _fit("log_murd_r_1ld")
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         coef2, se2, treat_var=treat_var, outcome_var="log_murd_r_1ld",
                         control_vars=CONTROLS, instrument_var=instrument_var, is_rct=False)

    ## Solution 3: effect on property crime
    coef3, se3 = _fit("log_prop_r_1ld")
    solution3 = Solution(id_li[2], title, query_li[2], method, dataset_name,
                         coef3, se3, treat_var=treat_var, outcome_var="log_prop_r_1ld",
                         control_vars=CONTROLS, instrument_var=instrument_var, is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2, id_li[2]: solution3}


def build_paper93(debug=False):
    """
    Builds the representation of paper 93

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Community and the crime decline: The causal effect of local nonprofits on violent crime"
    dataset_name = "sharkey_community"
    year = 2017
    domain = "criminology"
    n_solutions = 3

    query1 = "Does local nonprofit communities have a causal effect on violent crime?"
    query2 = "Does local nonprofit communities have a causal effect on murder rate?"
    query3 = "Does local nonprofit communities have a causal effect on property crime?"

    solutions = replicated_paper93(title, dataset_name, [query1, query2, query3], [160, 161, 162], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
