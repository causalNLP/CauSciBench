## This file contains the replication of paper 74
from pathlib import Path
from statsmodels.formula.api import mixedlm
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 74

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper74(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 74
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = 'post:osnap'
    control_vars = ['age', 'male', 'race', 'firstday']
    method = 'ols'

    ## Solution 1: effect of OSNAP on whole grain calorie consumption
    outcome_var1 = 's_wg_consumed'
    formula1 = 's_wg_consumed ~ post * osnap + age + male + C(race) + firstday'
    model1 = mixedlm(formula1, groups=df['childid'],
        vc_formula={'site': '0 + C(siteid)'}, data=df)
    result1 = model1.fit(reml=False)
    if debug:
        print(result1.summary())
    answer1  = result1.params[treat_var]
    std_err1 = result1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var1,
                         control_vars=control_vars, is_rct=True)

    ## Solution 2: effect of OSNAP on juice consumption
    outcome_var2 = 's_juiceoz_consumed'
    formula2 = 's_juiceoz_consumed ~ post * osnap + age + male + C(race) + firstday'
    model2 = mixedlm(formula2, groups=df['childid'],
        vc_formula={'site': '0 + C(siteid)'}, data=df)
    result2 = model2.fit(reml=False)
    if debug:
        print(result2.summary())
    answer2  = result2.params[treat_var]
    std_err2 = result2.bse[treat_var]
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treat_var, outcome_var=outcome_var2,
                         control_vars=control_vars, is_rct=True)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper74(debug=False):
    """
    Builds the representation of paper 74

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Impact of the out-of-school nutrition and physical activity (osnap) group randomized controlled trial on childrens food, beverage, and calorie consumption among snacks served"
    dataset_name = "lee_impact"
    year = 2018
    domain = "public health"
    n_solutions = 2

    query1 = "Does enrollment in OSNAP lead to an increase in the number of whole grain calories consumed by a child?"
    query2 = "Does enrollment in OSNAP lead to a reduction in the number of juices calories consumed by a child?"

    solutions = replicated_paper74(title, dataset_name,
                                   [query1, query2],
                                   [121, 122], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions, is_rct=True)
    return paper
