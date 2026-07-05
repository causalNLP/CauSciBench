## This file contains the replication of paper 64
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 64

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper64(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 64
    """

    method = 'did'
    state_var = 'state'
    time_var = 'year'

    ## Solution 1: effect of merit scholarships on college attendance
    df1 = pd.read_csv(BASE_DIR / "conley_inference_with_merit.csv")
    needed1 = ['coll', 'merit', 'male', 'black', 'asian', 'year', 'state']
    for c in needed1:
        df1[c] = pd.to_numeric(df1[c], errors='coerce')
    df1 = df1.dropna(subset=needed1).copy()
    df1['year']  = df1['year'].astype(int)
    df1['state'] = df1['state'].astype(int)

    treat_var1  = 'merit'
    outcome_var1 = 'coll'
    control_vars1 = ['male', 'black', 'asian']

    formula1 = 'coll ~ merit + male + black + asian + C(year) + C(state)'
    ols_fit1 = smf.ols(formula1, data=df1).fit()
    res_state1 = ols_fit1.get_robustcov_results(cov_type='cluster', groups=df1['state'])
    if debug:
        print(ols_fit1.summary())
    idx1     = list(ols_fit1.params.index).index(treat_var1)
    answer1  = float(res_state1.params[idx1])
    std_err1 = float(res_state1.bse[idx1])
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var1, outcome_var=outcome_var1,
                         control_vars=control_vars1, state_var=state_var,
                         time_var=time_var, is_rct=False)

    ## Solution 2: effect of HOPE scholarship on college attendance
    dataset2 = "conley_inference_with_hope"
    df2 = pd.read_csv(BASE_DIR / f"{dataset2}.csv")
    needed2 = ['collegeAttendance', 'hopeScholarship', 'male', 'black', 'asian', 'year', 'state']
    for c in needed2:
        df2[c] = pd.to_numeric(df2[c], errors='coerce')
    df2 = df2.dropna(subset=needed2).copy()
    df2['year']  = df2['year'].astype(int)
    df2['state'] = df2['state'].astype(int)

    treat_var2   = 'hopeScholarship'
    outcome_var2 = 'collegeAttendance'
    control_vars2 = ['male', 'black', 'asian']

    formula2 = 'collegeAttendance ~ hopeScholarship + male + black + asian + C(year) + C(state)'
    ols_fit2 = smf.ols(formula2, data=df2).fit()
    res_state2 = ols_fit2.get_robustcov_results(cov_type='cluster', groups=df2['state'])
    if debug:
        print(ols_fit2.summary())
    idx2     = list(ols_fit2.params.index).index(treat_var2)
    answer2  = float(res_state2.params[idx2])
    std_err2 = float(res_state2.bse[idx2])
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset2,
                         answer2, std_err2, treat_var=treat_var2, outcome_var=outcome_var2,
                         control_vars=control_vars2, state_var=state_var,
                         time_var=time_var, is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper64(debug=False):
    """
    Builds the representation of paper 64

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Inference with difference in differences with a small number of policy changes"
    dataset_name = "conley_inference_with_merit"
    year = 2005
    domain = "education"
    n_solutions = 2

    query1 = "What is the effect of the Merit scholarship on college attendance ?"
    query2 = "What is the effect of the HOPE scholarship on college attendance?"

    solutions = replicated_paper64(title, dataset_name,
                                   [query1, query2],
                                   [109, 110], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
