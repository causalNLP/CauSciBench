## This file contains the replication of paper 58
from pathlib import Path
import statsmodels.api as sm
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 58

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper58(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 58
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    required_nonmissing = ['SCYFNSH', 'FINISH6', 'PRSCHA_1', 'REPT6', 'NREPT', 'INSCHL',
        'FINISH7', 'VOUCH0', 'PRSCH_C', 'FINISH8', 'PRSCHA_2', 'TOTSCYRS', 'REPT']
    mask = (df['TAB3SMPL'] == 1)
    for var in required_nonmissing:
        mask &= df[var].notna()
    analytic = df.loc[mask].copy()

    month_cols = [c for c in analytic.columns if c.startswith('DMONTH')]
    strata_cols = [c for c in analytic.columns if c.startswith('STRATA')]
    use_month = [c for c in month_cols if c != 'DMONTH12'] if 'DMONTH12' in month_cols else month_cols[:-1]
    use_strata = [c for c in strata_cols if c != 'STRATA6'] if 'STRATA6' in strata_cols else strata_cols[:-1]

    treat_var = 'VOUCH0'
    basic_controls = ['SVY', 'HSVISIT', 'DJAMUNDI', 'PHONE', 'AGE', 'SEX2',
        'DBOGOTA', 'D1993', 'D1995', 'D1997'] + use_month + use_strata + ['SEX_MISS']
    method = 'ols'

    ## Solution 1: effect of voucher on marriage status
    outcome_var1 = 'MARRIED'
    d1 = analytic.dropna(subset=[treat_var, outcome_var1] + basic_controls)
    X1 = sm.add_constant(d1[[treat_var] + basic_controls].astype(float))
    y1 = d1[outcome_var1].astype(float)
    model1 = sm.OLS(y1, X1).fit(cov_type='HC0')
    if debug:
        print(model1.summary())
    answer1 = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var1,
                         control_vars=basic_controls, is_rct=True)

    ## Solution 2: effect of voucher on having a child
    outcome_var2 = 'HASCHILD'
    d2 = analytic.dropna(subset=[treat_var, outcome_var2] + basic_controls)
    X2 = sm.add_constant(d2[[treat_var] + basic_controls].astype(float))
    y2 = d2[outcome_var2].astype(float)
    model2 = sm.OLS(y2, X2).fit(cov_type='HC0')
    if debug:
        print(model2.summary())
    answer2 = model2.params[treat_var]
    std_err2 = model2.bse[treat_var]
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treat_var, outcome_var=outcome_var2,
                         control_vars=basic_controls, is_rct=True)

    ## Solution 3: effect of voucher on working status
    outcome_var3 = 'WORKING3'
    d3 = analytic.dropna(subset=[treat_var, outcome_var3] + basic_controls)
    X3 = sm.add_constant(d3[[treat_var] + basic_controls].astype(float))
    y3 = d3[outcome_var3].astype(float)
    model3 = sm.OLS(y3, X3).fit(cov_type='HC0')
    if debug:
        print(model3.summary())
    answer3 = model3.params[treat_var]
    std_err3 = model3.bse[treat_var]
    solution3 = Solution(id_li[2], title, query_li[2], method, dataset_name,
                         answer3, std_err3, treat_var=treat_var, outcome_var=outcome_var3,
                         control_vars=basic_controls, is_rct=True)

    ## Solution 4: effect of voucher on hours worked
    outcome_var4 = 'HOURSUM'
    d4 = analytic.dropna(subset=[treat_var, outcome_var4] + basic_controls)
    X4 = sm.add_constant(d4[[treat_var] + basic_controls].astype(float))
    y4 = d4[outcome_var4].astype(float)
    model4 = sm.OLS(y4, X4).fit(cov_type='HC0')
    if debug:
        print(model4.summary())
    answer4 = model4.params[treat_var]
    std_err4 = model4.bse[treat_var]
    solution4 = Solution(id_li[3], title, query_li[3], method, dataset_name,
                         answer4, std_err4, treat_var=treat_var, outcome_var=outcome_var4,
                         control_vars=basic_controls, is_rct=True)

    return {id_li[0]: solution1, id_li[1]: solution2,
            id_li[2]: solution3, id_li[3]: solution4}


def build_paper58(debug=False):
    """
    Builds the representation of paper 58

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Vouchers for private schooling in Colombia: Evidence from a randomized natural experiment"
    dataset_name = "angrist_vouchers"
    year = 2002
    domain = "education"
    is_rct = True
    is_multirct = False
    n_solutions = 4

    query1 = "Does winning a voucher affect whether students get married a few years later?"
    query2 = "Does winning a voucher reduce the likelihood of having a child?"
    query3 = "Does winning a lottery affect the likelihood of being in the labor force?"
    query4 = "Does winning a lottery affect the number of hours of work?"

    solutions = replicated_paper58(title, dataset_name,
                                   [query1, query2, query3, query4],
                                   [99, 100, 101, 102], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions,
                  is_multirct, is_rct, n_solutions)
    return paper
