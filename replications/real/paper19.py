## This file contains the replication of paper 19
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 19

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper19(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 19
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path)

    treat_var = "treat"
    outcome_var = "mm"
    state_var = "country"
    time_var = "year"
    control_vars = ["polity", "EcGI", "lgdp", "interwar", "intrawar", "lcinc", "efindex"]
    control_vars = [c for c in control_vars if c in df.columns]
    method = "did"

    work = df.dropna(subset=[outcome_var, treat_var, state_var, time_var] + control_vars)

    controls_str = " + ".join(control_vars)
    formula = f"{outcome_var} ~ {treat_var} + {controls_str} + C({state_var}) + C({time_var})"
    model = smf.ols(formula, data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work[state_var]})
    if debug:
        print(model.summary())
    answer1 = model.params[treat_var]
    std_err1 = model.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         canonical_did=False, is_rct=False)

    return {id_li[0]: solution1}


def build_paper19(debug=False):
    """
    Builds the representation of paper 19

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The causal effect of economic sanctions on political stability: A two-stage difference-in-differences analysis"
    dataset_name = "tan_causal_effect"
    year = 2024
    domain = "political science"
    n_solutions = 1

    query1 = "What is the effect of sanctions imposition on mass mobilization?"

    solutions = replicated_paper19(title, dataset_name, [query1], [32], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
