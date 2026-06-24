## This file contains the replication of paper 1
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 1


BASE_DIR = Path("../data/csv_files/realdata/")

def replicated_paper1(title, dataset_name, query_li, id_li, 
                      debug=False):
    """
    Replicates the analysis of paper 1
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path)

    treatment_var = "treatment"
    outcome_var = "voted"
    control_vars = ["g2000", "g2002", "p2000", "p2002", "p2004"]
    formula = f"{outcome_var} ~ C({treatment_var}, Treatment('Control')) + {' + '.join(control_vars)}"
    
    model = smf.ols(formula, data=df).fit()
    if debug:
        print(model.summary())
    method = "ols"

    key_param1 = f"C({treatment_var}, Treatment('Control'))[T.Hawthorne]"
    answer1 = model.params[key_param1]
    std_err1 = model.bse[key_param1]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name, 
                         answer1, std_err1, treat_var=treatment_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_multirct=True, is_rct=True)


    key_param2 = f"C({treatment_var}, Treatment('Control'))[T.Civic Duty]"
    answer2 = model.params[key_param2]
    std_err2 = model.bse[key_param2]

    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name, 
                         answer2, std_err2, treat_var=treatment_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_multirct=True, is_rct=True)

    key_param3 = f"C({treatment_var}, Treatment('Control'))[T.Neighbors]"
    answer3 = model.params[key_param3]
    std_err3 = model.bse[key_param3]
    solution3 = Solution(id_li[2], title, query_li[2], method, dataset_name, 
                         answer3, std_err3, treat_var=treatment_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_multirct=True, is_rct=True)



    key_param4 = f"C({treatment_var}, Treatment('Control'))[T.Self]"
    answer4 = model.params[key_param4]
    std_err4 = model.bse[key_param4]
    solution4 = Solution(id_li[3], title, query_li[3], method, dataset_name, 
                         answer4, std_err4, treat_var=treatment_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_multirct=True, is_rct=True)
    
    solution_dict = {id_li[0]: solution1, id_li[1]: solution2, 
                     id_li[2]: solution3, id_li[3]: solution4}
    
    return solution_dict

def build_paper1(debug=False):
    """
    Builds the representation of paper 1

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Social pressure and voter turnout: Evidence from a large-scale field experiment"
    dataset_name = "gerber_social_pressure"
    year = 2008
    domain = "political science"
    is_multirct = True
    is_rct = True
    n_solutions = 4

    query1 = "Does the Hawthorne scheme lead to an increase in voter turnout?"
    query2 = "Does reminding citizens about their Civic Duties make them more likely to vote?"
    query3 = "How effective is the Neighbors treatment scheme in increasing voter turnout?"
    query4 = "Is the Self-treatment method effective in increasing voter turnout?"

    solutions = replicated_paper1(title, dataset_name, [query1, query2, query3, query4], [1, 2, 3, 4], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, is_multirct,
                  is_rct, n_solutions)
    return paper




