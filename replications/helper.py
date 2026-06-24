## This file contains helper functions
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution
import numpy as np
from sklearn.linear_model import LogisticRegression
from dowhy import CausalModel as DoWhyCausalModel
from scipy import stats

def is_significant(answer, std_error, alpha=0.05):
    """
    Computes whether the estimated effect is statistically significant using a two-sided z-test.

    Args:
        answer (float): The estimated effect size
        std_error (float): The standard error of the estimate
        alpha (float): The significance level; the default is 0.05
    Returns:
        (int): 1 if the effect is statistically significant, 0 otherwise, None if inputs are invalid
    """
    if answer is None or std_error is None or std_error == 0:
        return None
    z = abs(answer / std_error)
    p_value = 2 * (1 - stats.norm.cdf(z))
    
    return int(p_value < alpha)


def diff_in_means(path, treat_var, outcome_var):
    """
    This function computes the difference in means (ATE) for a given dataset.

    Args:
        data_dir (Path): Path to the dataset
        treat_var (str): The name of the treatment variable; the default is "treatment"
        outcome_var (str): The name of the outcome variable; the default is "outcome"
    Returns:
       (float, float): Estimated effect and standard error
    """
    try:
        df = pd.read_csv(path)
        formula = f"{outcome_var} ~ {treat_var}"
        model = smf.ols(formula, data=df).fit()
        answer = model.params[treat_var]
        std_error = model.bse[treat_var]

        return (answer, std_error)

    except Exception as e:
        print(f"Error reading {path}: {e}")


def replicate_ihdp(base_folder, common_name, treat_var="treatment", outcome_var="y", n_data=10, 
                   control_vars=None):
    """
    This function results the solution for the IHDP dataset in QRData. 
    Since the data is from a randomized trial, we use difference in means. 

    Args:
        base_folder (str): The base folder where the dataset is located 
        common_name (str): The common name of the dataset; the default naming is ihdp_1, ihdp_2,..
        treat_var (str): The name of the treatment variable; the default is "treatment"
        outcome_var (str): The name of the outcome variable; the default is "outcome"
        n_data (int): The number of IHDP data 
    Returns:
        (dict): A dictionary mapping (int) dataset index -> (float, float) effect and standard error
    """

    all_solutions = {}

    for i in range(n_data):
        full_path = f"{base_folder}/{common_name}_{i}.csv"
        formula = f"{outcome_var} ~ {treat_var}"
        if control_vars:
            formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)}"
        
        model = smf.ols(formula, data=pd.read_csv(full_path)).fit()
        answer = model.params[treat_var]
        std = model.bse[treat_var]
        all_solutions[i] = (answer, std)
    
    return all_solutions

def build_ihdp_solution(base_folder="../data/csv_files/qrdata", common_name="ihdp", treat_var="treatment",
                   outcome_var="y", n_data=10):
    """
    This function builds the solution for the QRData dataset.

    Args:
        base_folder (str): The base folder where the dataset is located 
        common_name (str): The common name of the dataset; the default naming is ihdp_1, ihdp_2,..
        treat_var (str): The name of the treatment variable; the default is "treatment"
        outcome_var (str): The name of the outcome variable; the default is "outcome"
        n_data (int): The number of IHDP data 
    Returns:
        (dict): A dictionary mapping (int) dataset index -> (Solution) solution to the problem
    """

    paper_name = "QRData"
    query0 = "Does home visit from specialist doctors lead to an improvement in cognitive scores?"
    query_rest = "What is the effect of home visits on the cognitive test scores of children who" \
                  " actually received the intervention?"
    method = "ols"
    is_rct = True

    controls = [f"x{i}" for i in range(1, 26)]
    numerical_vals = replicate_ihdp(base_folder, common_name, treat_var, outcome_var, n_data)
    all_solutions = {}
    all_solutions[0] = Solution(1, paper_name, query0, method, "ihdp_0.csv",
                                numerical_vals[0][0], numerical_vals[0][1],
                                treat_var=treat_var, outcome_var=outcome_var, is_rct=is_rct, 
                                control_vars=controls)
    for i in range(1, n_data):
        answer, std = numerical_vals[i]

        all_solutions[i] = Solution(i+1, paper_name, query_rest, method,
                                    f"ihdp_{i}.csv", answer, std, control_vars=controls,
                                    treat_var=treat_var, outcome_var=outcome_var, is_rct=is_rct)
        
    
    return all_solutions

def compute_att_matching_manual(df, treatment, outcome, confounders, matches=1):
    """
    computes the average treatment effect on the treated group (ATT) using matching;
    We manually compute the porpoensity scores. 

    Args:
        df (pd.DataFrame): The dataset
        treatment (str): The name of the treatment variable
        outcome (str): The name of the outcome variable
        causes (list): A list of the names of the confounding variables
        matches (int): The number of (control) units to match for each treated unit
    Returns:
        (float, float): The average treatment effect and its standard error
    """

    X = df[confounders].values
    T = df[treatment].values
    Y = df[outcome].values

    ps = LogisticRegression(max_iter=5000).fit(X, T).predict_proba(X)[:, 1]

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    ps_control = ps[control_idx]

    individual_effects = []
    for i in treated_idx:
        nearest = np.argsort(np.abs(ps_control - ps[i]))[:matches]
        individual_effects.append(Y[i] - Y[control_idx[nearest]].mean())

    att = np.mean(individual_effects)
    std_error = np.std(individual_effects, ddof=1) / np.sqrt(len(individual_effects))

    return att, std_error


def compute_att_dowhy(df, treatment, outcome, confounders, estimand="att"):
    """
    Computes the average treatment effect on the treated group (ATT) using the DoWhy library.

    Args:
        df (pd.DataFrame): The dataset
        treatment (str): The name of the treatment variable
        outcome (str): The name of the outcome variable
        confounders (list): A list of the names of the confounding variables
        estimand (str): The type of estimand to compute; the default is "att"
    Returns:
        (float, float): The average treatment effect and its standard error
    """

    model = DoWhyCausalModel(data=df, treatment=treatment, outcome=outcome, 
                             common_causes=confounders)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    att = model.estimate_effect(identified_estimand, target_units="att",
                                     method_name="backdoor.propensity_score_matching")
    
    return att.value, att.get_standard_error()


def build_jobs_solution(base_folder="../data/csv_files/qrdata", common_name="jobs", treat_var="t",
                        outcome_var="y",confounders=[f"x{i}" for i in range(0, 17)], n_data=10):
    """
    Builds the solution for the jobs dataset

    Returns:
        dict: A dictionary containing the solutions for each dataset: (int) index -> (Solution) solution
    """

    paper_name = "QRData"
    query = "Does the job training program improve employment prospects?"
    treat_var = treat_var
    outcome_var = outcome_var
    control_vars = confounders
    method = "matching"
    all_solutions = {}

    ## the values are hard-coded
    n_units = [2, 2, 4, 4, 1, 2, 1, 3, 1, 4]
    start = 30

    for i in range(n_data):
        df = pd.read_csv(f"{base_folder}/{common_name}_{i}.csv")
        att, std_error = compute_att_matching_manual(df, treat_var, outcome_var, control_vars, 
                                                     n_units[i])
        if i == 4:
            att, std_error = compute_att_dowhy(df, treat_var, outcome_var, control_vars)
        all_solutions[i] = Solution(start + i,paper_name, query, method,
                                    f"jobs_{i}.csv", att, std_error, treat_var=treat_var, 
                                    outcome_var=outcome_var,control_vars=control_vars)
    return all_solutions



                        
    
