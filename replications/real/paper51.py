## This file contains the replication of paper 51
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 51

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper51(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 51
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "euvote"
    outcome_var = "introad"
    control_vars = ["tab", "timegdk_98", "neighbourban", "puop", "regcul",
                    "leftregperc", "gerpop", "pop2", "t1", "t1_2", "t1_3"]
    method = "glm"

    df_clean = df.dropna(subset=[outcome_var, treat_var] + control_vars + ["canton"])

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)}"
    res = smf.logit(formula, data=df_clean).fit(
        cov_type="cluster", cov_kwds={"groups": df_clean["canton"]}, disp=0)

    if debug:
        print(res.summary())

    answer = res.params[treat_var]
    std_err = res.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper51(debug=False):
    """
    Builds the representation of paper 51

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Europeanisation beyond the EU: Tobacco advertisement restrictions in Swiss cantons"
    dataset_name = "trein_europeanisation"
    year = 2017
    domain = "political science"
    n_solutions = 1

    query1 = "How does support for European integration affect the probability that a canton adopts a tobacco advertisement restriction this year?"

    solutions = replicated_paper51(title, dataset_name, [query1], [90], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
