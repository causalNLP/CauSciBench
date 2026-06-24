## This file contains the replication of paper 54
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 54

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper54(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 54
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    if "time" not in df.columns:
        df["time"] = (df["year"] == 2010).astype(int)

    nat_city_names = {"Hà N?i", "TP.H? Chí Minh", "H?i Phòng", "Đà Nẵng",
                      "Cần Thơ", "?à N?ng", "C?n Th?"}
    df["city"] = df["tentinh"].isin(nat_city_names).astype(int)

    treat_var = "treatment"
    method = "did"
    control_vars = ["time", "lnarea", "lnpopden", "city"]
    state_var = "reg8"
    time_var = "time"

    outcome_var = "transport"
    need = [outcome_var, "time", treat_var, "lnarea", "lnpopden", "city", "reg8"]
    use = df.dropna(subset=need).copy()
    use["reg8"] = use["reg8"].astype(int).astype(str)

    formula = f"{outcome_var} ~ time + {treat_var} + time:{treat_var} + lnarea + lnpopden + city + C(reg8)"
    res = smf.ols(formula, data=use).fit()

    if debug:
        print(res.summary())

    answer = res.params[f"time:{treat_var}"]
    std_err = res.bse[f"time:{treat_var}"]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         is_rct=False)

    return {id_li[0]: solution1}


def build_paper54(debug=False):
    """
    Builds the representation of paper 54

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The impact of recentralization on public services: A difference-in-differences analysis of the abolition of elected councils in Vietnam"
    dataset_name = "malesky_impact_of_recentralization"
    year = 2013
    domain = "public economics"
    n_solutions = 1

    query1 = "What is the effect of the 2009 recentralization on whether a commune has public transport?"

    solutions = replicated_paper54(title, dataset_name, [query1], [93], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
