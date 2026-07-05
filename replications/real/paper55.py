## This file contains the replication of paper 55
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from solution import Solution, Paper

PAPER_ID = 55

BASE_DIR = Path("../data/csv_files/realdata/")

BINARY_VARS = ["emotionalsupport", "caresupport", "economicpoverty", "healthpoverty",
               "rightspoverty", "spiritualpoverty", "multidimensionalpoverty",
               "gender", "maritalstatus", "householdregistration", "socialsecurity"]


def _coerce_numeric(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_and_binarize(dataset_name):
    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    for v in BINARY_VARS:
        if v in df.columns:
            df[v] = (df[v] > 0).astype(int)
    return df


def _run_glm(d, outcome_var, treat_var):
    d = pd.concat([d.drop(columns=["region"]),
                   pd.get_dummies(d["region"].astype(str), prefix="region", drop_first=True)], axis=1)
    d = _coerce_numeric(d).replace([np.inf, -np.inf], np.nan).dropna()
    zero_var = [c for c in d.columns if c != outcome_var and d[c].nunique() <= 1]
    if zero_var:
        d = d.drop(columns=zero_var)
    y = d[outcome_var].astype(int)
    X = sm.add_constant(d.drop(columns=[outcome_var]).astype(float), has_constant="add")
    glm = sm.GLM(y.values.astype(float), X.values.astype(float), family=sm.families.Binomial())
    res = glm.fit(cov_type="HC1")
    idx = X.columns.get_loc(treat_var)
    return float(res.params[idx]), float(res.bse[idx]), int(res.nobs)


def _query1(df):
    controls = ["caresupport", "emotionalsupport", "gender", "age", "squareofage", "maritalstatus",
                "householdregistration", "totalnumberofchildren", "proportionofboys",
                "numberofpeoplelivingtogether", "totalhouseholdincome", "socialsecurity", "region"]
    use = ["spiritualpoverty", "economicsupport"] + controls
    d = df[use].dropna().copy()
    return _run_glm(d, "spiritualpoverty", "economicsupport")


def _query2(df):
    controls = ["socialsecurity", "gender", "age", "squareofage", "maritalstatus",
                "householdregistration", "totalnumberofchildren", "numberofboys",
                "proportionofboys", "totalhouseholdincome", "numberofpeoplelivingtogether", "region"]
    use = ["multidimensionalpoverty", "emotionalsupport"] + [c for c in controls if c in df.columns]
    d = df[use].dropna().copy()
    return _run_glm(d, "multidimensionalpoverty", "emotionalsupport")


def _query3(df):
    controls = ["economicsupport", "emotionalsupport", "gender", "age", "squareofage", "maritalstatus",
                "householdregistration", "totalnumberofchildren", "proportionofboys",
                "numberofpeoplelivingtogether", "totalhouseholdincome", "socialsecurity", "region"]
    use = ["rightspoverty", "caresupport"] + controls
    d = df[use].dropna().copy()
    return _run_glm(d, "rightspoverty", "caresupport")


def replicated_paper55(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 55
    """

    df = _load_and_binarize(dataset_name)
    method = "glm"

    query_fns = [_query1, _query2, _query3]
    treat_vars = ["economicsupport", "emotionalsupport", "caresupport"]
    outcome_vars = ["spiritualpoverty", "multidimensionalpoverty", "rightspoverty"]

    solutions = {}
    for i, (fn, treat_var, outcome_var) in enumerate(zip(query_fns, treat_vars, outcome_vars)):
        answer, std_err, n = fn(df)
        if debug:
            print(f"Q{i+1} ({outcome_var}): {answer:.4f} ({std_err:.4f}), N={n}")
        solutions[id_li[i]] = Solution(id_li[i], title, query_li[i], method, dataset_name,
                                       answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                                       is_rct=False)

    return solutions


def build_paper55(debug=False):
    """
    Builds the representation of paper 55

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The impact of intergenerational support on multidimensional poverty in old age: Empirical analysis based on 2018 CLHLS data"
    dataset_name = "tan_impact_of_intergenerational"
    year = 2023
    domain = "economics"
    n_solutions = 3

    query1="How does receiving economic support affect spiritual poverty?"
    query2="What is the effect of receiving emotional support on multidimensional poverty?"
    query3="How does receiving care support influence rights poverty?"

    queries = [query1, query2, query3]

    solutions = replicated_paper55(title, dataset_name, queries, [94, 95, 96], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
