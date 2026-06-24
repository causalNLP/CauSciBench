## This file contains the solutions related to queries in QRData

from helper import build_ihdp_solution, build_jobs_solution
from solution import Solution

import pandas as pd
import numpy as np 
import statsmodels.formula.api as smf

from linearmodels.iv import IV2SLS
from causalinference import CausalModel
from dowhy import CausalModel as DoWhyCausalModel
from pathlib import Path 


_ihdp_solutions = build_ihdp_solution()
_jobs_solutions = build_jobs_solution()
BASE_FOLDER_PATH = Path("../data/csv_files/qrdata")
def query_1():
    """
    This function returns the solution for query 1 in Solution class representation
    """

    return _ihdp_solutions[0]

def query_2():
    """
    This function returns the solution for query 2 in Solution class representation
    """
    return _ihdp_solutions[1]

def query_3():
    """
    This function returns the solution for query 3 in Solution class representation
    """
    return _ihdp_solutions[2]

def query_4():
    """
    This function returns the solution for query 4 in Solution class representation
    """
    return _ihdp_solutions[3]

def query_5():
    """
    This function returns the solution for query 5 in Solution class representation
    """                     
    return _ihdp_solutions[4]

def query_6():
    """
    This function returns the solution for query 6 in Solution class representation
    """

    return _ihdp_solutions[5]

def query_7():
    """
    This function returns the solution for query 7 in Solution class representation
    """

    return _ihdp_solutions[6]

def query_8():
    """
    This function returns the solution for query 8 in Solution class representation
    """

    return _ihdp_solutions[7]

def query_9():
    """
    This function returns the solution for query 9 in Solution class representation
    """

    return _ihdp_solutions[8]

def query_10():
    """
    This function returns the solution for query 10 in Solution class representation
    """

    return _ihdp_solutions[9]


def query_11():
    """
    This function returns the solution for query 11 in Solution class representation
    """

    paper_name = "QRData"
    name = "online_classroom.csv"
    dataset = BASE_FOLDER_PATH / name
    df = pd.read_csv(dataset).query("format_blended == 0")
    outcome_var = "falsexam"
    treat_var = "format_ol"
    query = "Is there any advantage to taking classes online in comparison to" \
             " face-to-face or blended format in terms of exam performance? "
    method = "ols"

    model = smf.ols(f'{outcome_var} ~ {treat_var}', data=df).fit()
    answer = model.params[treat_var]
    std_error = model.bse[treat_var]


    solution = Solution(11, paper_name, query, method, name, answer, std_error,
                        treat_var=treat_var, outcome_var=outcome_var, is_rct=True)
    
    return solution

def query_12():
    """
    This function returns the solution for query 25 in Solution class representation
    """

    paper_name = "QRData"
    name = "wage.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Do more years of education lead to higher log hourly wages?"
    method = "ols"
    outcome_var = "hwage"
    treat_var = "educ"

    formula = f"np.log({outcome_var}) ~ {treat_var}"

    data = pd.read_csv(dataset)
    model = smf.ols(formula, data=data).fit()
    answer = model.params[treat_var]
    std_error = model.bse[treat_var]

    return Solution(12,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var, is_rct=False)


def query_13():
    """
    This function returns the solution for query 24 in Solution class representation
    """

    paper_name = "QRData"
    name = "wage.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Does an increase in years of education help increase log hourly "\
             "wages when also considering other variables of interest?"
    method = "ols"
    outcome_var = "hwage"
    treat_var = "educ"

    controls = ['IQ', 'exper', 'tenure', 'age', 'married', 'black',
            'south', 'urban', 'sibs', 'brthord', 'meduc', 'feduc']
    formula = f"np.log({outcome_var}) ~ {treat_var} + {' + '.join(controls)}"

    data = pd.read_csv(dataset)
    model = smf.ols(formula, data=data).fit()
    answer = model.params[treat_var]
    std_error = model.bse[treat_var]

    return Solution(13,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var, is_rct=False, 
                    control_vars=controls)

def query_14():
    """
    This function returns the solution for query 14 in Solution class representation
    """

    paper_name = "QRData"
    name = "wage.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "What's the effect of graduating 12th grade on hourly wage?"
    method = "ols"
    outcome_var = "hwage"
    treat_var = "passed_12"

    formula = f"{outcome_var} ~ {treat_var}"
    data = pd.read_csv(dataset)
    model = smf.ols(formula, data=data).fit()
    answer = model.params[treat_var]
    std_error = model.bse[treat_var]

    return Solution(14, paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var)

def query_15():
    """
    This function returns the solution for query 15 in Solution class representation
    """

    paper_name = "QRData"
    name = "collections_email.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Does sending reminder emails have an impact on the repayment of debts"\
             "while including suitable controls?"
    method = "ols"
    treat_var = "email"
    outcome_var = "payments"
    control_vars = ["credit_limit", "risk_score"]

    data = pd.read_csv(dataset)
    model = smf.ols(f'{outcome_var} ~ {treat_var} + credit_limit + risk_score', data=data).fit()
    answer = model.params[treat_var]
    std_error = model.bse[treat_var]

    return Solution(15, paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    control_vars=control_vars, is_rct=True)


def query_16():
    """
    This function returns the solution for query 16 in Solution class representation
    """

    paper_name = "QRData"
    name = "hospital_treatment.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Is the new drug effective in reducing the hospital stay?"
    method = "ols"
    treat_var = "treatment"
    outcome_var = "days"
    control_vars = ["severity"]

    data = pd.read_csv(dataset)
    model = smf.ols(f'{outcome_var} ~ {treat_var} + severity', data=data).fit()
    answer = model.params[treat_var]
    std_error = model.bse[treat_var]

    return Solution(16, paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    control_vars=control_vars, is_rct=True)


def query_17():
    """
    This function returns the solution for query 17 in Solution class representation
    """

    paper_name = "QRData"
    name = "ak91.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "How much more does a person earn for each extra year of education?"
    method = "iv"
    treat_var = "years_of_schooling"
    outcome_var = "log_wage"
    instrument_var = "q4"
    control_vars = ["year_of_birth", "state_of_birth"]

    data = pd.read_csv(dataset)
    data["q4"] = (data["quarter_of_birth"] == 4.0).astype(int)

    formula = f"log_wage ~ 1 + C(year_of_birth) + C(state_of_birth) + [years_of_schooling ~ q4]"
    model = IV2SLS.from_formula(formula, data).fit()
    answer = model.params[treat_var]
    std_error = model.std_errors[treat_var]

    return Solution(17,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    control_vars=control_vars, instrument_var=instrument_var)


def query_18():
    """
    This function returns the solution for query 18 in Solution class representation
    """

    paper_name = "QRData"
    name = "app_engagement_push.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Did the marketing push help increase in-app purchases?"
    method = "iv"
    treat_var = "push_delivered"
    outcome_var = "in_app_purchase"
    instrument_var = "push_assigned"

    data = pd.read_csv(dataset)
    model = IV2SLS.from_formula(f"{outcome_var} ~ 1 + [{treat_var} ~ {instrument_var}]", data).fit()
    answer = model.params[treat_var]
    std_error = model.std_errors[treat_var]

    return Solution(18,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    instrument_var=instrument_var, is_rct=True)


def query_19():
    """
    This function returns the solution for query 19 in Solution class representation
    """

    paper_name = "QRData"
    name = "medicine_impact_recovery.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "What is the effect of the medication on the recovery time?"
    method = "matching"
    treat_var = "medication"
    outcome_var = "recovery"
    control_vars = ["severity", "age", "sex"]

    data = pd.read_csv(dataset)
    cm = CausalModel(Y=data[outcome_var].values, D=data[treat_var].values,
                     X=data[control_vars].values)
    cm.est_via_matching(matches=1, bias_adj=True)
    answer = cm.estimates["matching"]["ate"]
    std_error = cm.estimates["matching"]["ate_se"]

    return Solution(19,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    control_vars=control_vars, is_rct=False)

def query_20():
    """
    This function returns the solution for query 20 in Solution class representation
    """

    paper_name = "QRData"
    name = "learning_mindset.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Does participating in seminars meant for boosting growth mindset "\
             "lead to better academic achievement?"
    method = "ipw"
    treat_var = "intervention"
    outcome_var = "achievement_score"
    causes = ["ethnicity", "gender", "school_urbanicity"]

    data = pd.read_csv(dataset)
    data_with_categ = pd.concat([data.drop(columns=causes), 
                                 pd.get_dummies(data[causes], columns=causes, drop_first=False)],
                                 axis=1)
    control_vars = [col for col in data_with_categ.columns
                    if col not in ("schoolid", treat_var, outcome_var)]

    model = DoWhyCausalModel(data=data_with_categ, treatment=treat_var, outcome=outcome_var,
                             common_causes=control_vars)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    ate = model.estimate_effect(identified_estimand, target_units="ate",
                                method_name="backdoor.propensity_score_matching",
                                method_params={"weighting_scheme": "ips_weight"})
    answer = ate.value
    std_error = ate.get_standard_error()

    return Solution(20,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    control_vars=control_vars, is_rct=False)


def query_21():
    """
    This function returns the solution for query 21 in Solution class representation
    """

    paper_name = "QRData"
    name = "billboard_impact.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Does using billboards lead to higher bank deposits?"
    method = "did"
    treat_var = "poa"
    outcome_var = "deposits"
    time_var = "jul"
    interaction_var = "poa:jul"

    data = pd.read_csv(dataset)
    model = smf.ols(f"{outcome_var} ~ {treat_var}*{time_var}", data=data).fit()
    answer = model.params[interaction_var]
    std_error = model.bse[interaction_var]

    return Solution(21,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    time_var=time_var, interaction_var=interaction_var, canonical_did=True, is_rct=False)



def query_22():
    """
    This function returns the solution for query 22 in Solution class representation
    """

    paper_name = "QRData"
    name = "drinking.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "By how much does turning 21, the legal drinking age affect"\
            " the risk of death from any cause?"
    method = "rdd"
    outcome_var = "all"
    running_var = "agecell"
    cutoff = 21

    data = pd.read_csv(dataset)
    data["running"] = data[running_var] - cutoff
    data["threshold"] = data["running"] > 0

    model = smf.wls(f"{outcome_var} ~ running*threshold", data=data).fit()
    answer = model.params["threshold[T.True]"]
    std_error = model.bse["threshold[T.True]"]

    return Solution(22,paper_name, query, method, name, answer, std_error,
                    outcome_var=outcome_var, running_var=running_var, is_rct=False)


def query_23():
    """
    This function returns the solution for query 23 in Solution class representation
    """

    paper_name = "QRData"
    name = "smoking2.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Did Proposition 99 help reduce cigarette sales?"
    method = "did"

    treat_var = "california"
    outcome_var = "cigsale"
    time_var = "after_treatment"
    did_term = "california:after_treatment"

    data = pd.read_csv(dataset)
    data[treat_var] = data[treat_var].astype(int)
    data[time_var] = data[time_var].astype(int)
    model = smf.ols(f"{outcome_var} ~ {treat_var}*{time_var}", data=data).fit()
    answer = model.params[did_term]
    std_error = model.bse[did_term]

    return Solution(23,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    time_var=time_var, canonical_did=True)

def query_24():
    """
    This function returns the solution for query 24 in Solution class representation
    """

    paper_name = "QRData"
    name = "trainee_unique_on_age.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "How much more did people who joined the trainee program earn compared"\
             "to similar people who didn’t?"
    method = "matching"
    treat_var = "trainees"
    outcome_var = "earnings"
    control_vars = ["age"]

    data = pd.read_csv(dataset)
    cm = DoWhyCausalModel(data=data, treatment=treat_var, outcome=outcome_var,
                          common_causes=control_vars)
    estimand = cm.identify_effect(proceed_when_unidentifiable=True)
    att = cm.estimate_effect(estimand, target_units="att",
                             method_name="backdoor.propensity_score_matching")
    answer = att.value
    std_error = att.get_standard_error()

    return Solution(24,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    control_vars=control_vars, is_rct=False)


def query_25():
    """
    This function returns the solution for query 25 in Solution class representation
    """

    paper_name = "QRData"
    name = "MPs.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "What is the effect of becoming members of Parliament on the net (log) wealth"\
            "for Tory candidates?" 
    method = "rdd"
    outcome_var = "net"
    running_var = "margin"
    cutoff= 0

    data = pd.read_csv(dataset)
    tory = data[data["party"] == "tory"]

    tory = tory.copy()
    tory["threshold"] = (tory[running_var] > cutoff).astype(int)
    model = smf.ols(f"{outcome_var} ~ {running_var} * threshold", data=tory).fit()
    answer = model.params["threshold"]
    std_error = model.bse["threshold"]

    return Solution(25,paper_name, query, method, name, answer, std_error,
                    outcome_var=outcome_var, running_var=running_var, is_rct=False)


def query_26():
    """
    This function returns the solution for query 26 in Solution class representation
    """

    paper_name = "QRData"
    name = "women.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "Did the reservation policy lead to more new or repaired drinking water facilities in villages since it was implemented?"
    method = "ols"
    treat_var = "reserved"
    outcome_var = "water"

    data = pd.read_csv(dataset)
    model = smf.ols(f"{outcome_var} ~ {treat_var}", data=data).fit()
    answer = model.params[treat_var]
    std_error = model.bse[treat_var]

    return Solution(26,paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

def query_27():
    """
    This function returns the solution for query 27 in Solution class representation
    """

    paper_name = "QRData"
    name = "social.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "What is the difference in the effect of the Neighbors message on whether the voter voted in the 2006"\
            " primary election between those who voted in the 2004 primary election and those who did not?"
    method = "ols"
    treat_var = "messages"
    outcome_var = "primary2006"
    interaction_term = "primary2004"
    ## this is a multi-treatment setting 
    formula_tern = f"{interaction_term} * C({treat_var}, Treatment(reference='Control'))"
    key_param = f"{interaction_term}:C({treat_var}, Treatment(reference='Control'))[T.Neighbors]"

    data = pd.read_csv(dataset)
    model = smf.ols(f"{outcome_var} ~ {formula_tern}", data=data).fit()
    answer = model.params[key_param]
    std_error = model.bse[key_param]

    return Solution(27, paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    is_rct=True, interaction_var=interaction_term, is_multirct=True)


def query_28():
    """
    This function returns the solution for query 28 in Solution class representation
    """

    paper_name = "QRData"
    name = "social.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "What is the effect of the Neighbors message on whether the voter voted in the 2006 primary "\
            "election if the voter's age was 25 in 2006?"
    method = "ols"
    treat_var = "messages"
    outcome_var = "primary2006"
    interaction_term = "age"

    ## this is a multi-treatment setting 
    key_param = f"C({treat_var}, Treatment(reference='Control'))[T.Neighbors]:C({interaction_term})[T.25]" 
    formula = f"{outcome_var} ~ C({treat_var}, Treatment(reference='Control')) * C({interaction_term})"
    data = pd.read_csv(dataset)
    model = smf.ols(formula, data=data).fit()
    #(model.summary())
    answer = model.params[key_param]
    std_error = model.bse[key_param]

    return Solution(28, paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    interaction_var=interaction_term, is_rct=True, is_multirct=True)

def query_29():
    """
    This function returns the solution for query 28 in Solution class representation
    """

    paper_name = "QRData"
    name = "social.csv"
    dataset = BASE_FOLDER_PATH / name
    query = "What is the effect of the Neighbors message on whether the voter voted in the 2006 primary "\
            "election if the voter's age was 65 in 2006?"
    method = "ols"
    treat_var = "messages"
    outcome_var = "primary2006"
    interaction_term = "age"

    ## this is a multi-treatment setting 
    key_param = f"C({treat_var}, Treatment(reference='Control'))[T.Neighbors]:C({interaction_term})[T.65]" 
    formula = f"{outcome_var} ~ C({treat_var}, Treatment(reference='Control')) * C({interaction_term})"
    data = pd.read_csv(dataset)
    model = smf.ols(formula, data=data).fit()
    answer = model.params[key_param]
    std_error = model.bse[key_param]

    return Solution(29, paper_name, query, method, name, answer, std_error,
                    treat_var=treat_var, outcome_var=outcome_var,
                    interaction_var=interaction_term, is_rct=True, is_multirct=True)

def query_30():
    """
    This function returns the solution for query 30 in Solution class representation. 
    """

    return _jobs_solutions[0]

def query_31():
    """
    This function returns the solution for query 31 in Solution class representation. 
    """

    return _jobs_solutions[1]

def query_32():
    """
    This function returns the solution for query 32 in Solution class representation. 
    """

    return _jobs_solutions[2]

def query_33():
    """
    This function returns the solution for query 33 in Solution class representation. 
    """

    return _jobs_solutions[3]

def query_34():
    """
    This function returns the solution for query 34 in Solution class representation. 
    """

    return _jobs_solutions[4]

def query_35():
    """
    This function returns the solution for query 35 in Solution class representation. 
    """

    return _jobs_solutions[5]

def query_36():
    """
    This function returns the solution for query 36 in Solution class representation. 
    """

    return _jobs_solutions[6]

def query_37():
    """
    This function returns the solution for query 37 in Solution class representation. 
    """

    return _jobs_solutions[7]

def query_38():
    """
    This function returns the solution for query 38 in Solution class representation. 
    """

    return _jobs_solutions[8]

def query_39():
    """
    This function returns the solution for query 39 in Solution class representation.
    """

    return _jobs_solutions[9]


def solve_all():
    """
    Invokes all query functions and collects all solutions.

    Returns:
        (dict): A dictionary mapping (int) query id -> (Solution) solution
    """
    query_fns = [query_1, query_2, query_3, query_4, query_5, query_6, query_7, 
                 query_8, query_9, query_10, query_11, query_12, query_13, query_14, 
                 query_15, query_16, query_17, query_18, query_19, query_20, query_21, 
                 query_22, query_23, query_24, query_25, query_26, query_27, query_28, 
                 query_29, query_30, query_31, query_32, query_33, query_34, query_35,
                 query_36, query_37, query_38, query_39]
    
    solutions = [fn() for fn in query_fns]
    
    return {s.id: s for s in solutions}


