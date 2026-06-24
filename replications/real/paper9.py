## This file contains the replication of paper 9
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 9

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper9(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 9
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    treat_var = "cdl"
    state_var = "sid"
    time_var = "year"
    method = "did"
    outcome_vars = ["l_homicide",  "l_motor", "l_larceny", "l_burglary"]

    solutions = {}

    for i, outcome_var in enumerate(outcome_vars):
        formula = f"{outcome_var} ~ {treat_var} + C({state_var}) + C({time_var})"
        # larceny is weighted by population size
        if outcome_var == "l_larceny":
            model = smf.wls(formula, data=df, weights=df["popwt"]).fit(
                cov_type="cluster", cov_kwds={"groups": df[state_var]})
        else:
            model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[state_var]})
        if debug:
            print(f"\n--- {outcome_var} ---")
            print(model.summary())
        answer = model.params[treat_var]
        std_err = model.bse[treat_var]
        solution = Solution(id_li[i], title, query_li[i], method, dataset_name,
                            answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                            state_var=state_var, time_var=time_var,
                            canonical_did=False, is_rct=False)
        solutions[id_li[i]] = solution

    return solutions


def build_paper9(debug=False):
    """
    Builds the representation of paper 9

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Does strengthening self-defense law deter crime or escalate violence?"
    dataset_name = "cheng_does_strengthening"
    year = 2013
    domain = "law"
    n_solutions = 4

    query1 = "Did the castle doctrine laws lead to a reduction in (log) homicide rates compared to states that did not adopt the law?"
    query2 = "What is the effect of castle doctrine laws on (log) motor vehicle theft rates?"
    query3 = "How do castle doctrine laws affect (log) larceny rates across states while accounting for population size across the states?"
    query4 = "What is the effect of castle doctrine laws on (log) burglary?"

    solutions = replicated_paper9(title, dataset_name,
                                  [query1, query2, query3, query4],
                                  [15, 16, 17, 18], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
