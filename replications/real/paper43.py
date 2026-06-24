## This file contains the replication of paper 43
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 43

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper43(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 43
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    outcome_var = "vtr_gen05"
    method = "iv"

    df["treatment"] = 0
    df.loc[df["control"] == 0, "treatment"] = 1
    df.loc[df["control"] == 1, "treatment"] = 0

    solutions = {}

    treat_var1 = "treated"
    instrument_var1 = "treatment"

    data1 = df[[outcome_var, treat_var1, instrument_var1]].dropna()
    exog1 = sm.add_constant(data1[[]])
    res1 = IV2SLS(data1[outcome_var], exog1, data1[[treat_var1]], data1[[instrument_var1]]).fit()

    if debug:
        print(res1.summary)

    solutions[id_li[0]] = Solution(id_li[0], title, query_li[0], method, dataset_name,
                                   res1.params[treat_var1], res1.std_errors[treat_var1],
                                   treat_var=treat_var1, outcome_var=outcome_var,
                                   instrument_var=instrument_var1, is_rct=True)

    endog_vars = ["contacted_4weeks", "contacted_2weeks", "contacted_3days"]
    instr_vars = ["four_weeks", "two_weeks", "three_days"]
    control_vars = ["democrat", "republican", "unenrolled", "other_party",
                    "age", "age_squared", "vtr_gen01", "vtr_gen03"]

    df["high"] = (df["vote_prop"] > 0.68).astype(int)

    df_use = df[df["use"] == 1]
    df_high = df_use[df_use["high"] == 1]

    df_low = df_use[df_use["high"] == 0]

    for subset, offset in [(df_use, 1), (df_high, 4)]:
        for i, (ev, iv) in enumerate(zip(endog_vars, instr_vars)):
            cols = [outcome_var, ev] + control_vars + [iv]
            data = subset[cols].dropna()
            exog = sm.add_constant(data[control_vars])
            res = IV2SLS(data[outcome_var], exog, data[[ev]], data[[iv]]).fit()

            if debug:
                print(f"{ev} (offset {offset}): {res.params[ev]:.4f} ({res.std_errors[ev]:.4f})")

            idx = id_li[offset + i]
            solutions[idx] = Solution(idx, title, query_li[offset + i],
                                      method, dataset_name,
                                      res.params[ev], res.std_errors[ev],
                                      treat_var=ev, outcome_var=outcome_var,
                                      control_vars=control_vars, instrument_var=iv, is_rct=True)

    # Queries 8–9: low-salience (high==0), 4 weeks and 3 days only
    for i, (ev, iv) in enumerate([("contacted_4weeks", "four_weeks"), ("contacted_3days", "three_days")]):
        cols = [outcome_var, ev] + control_vars + [iv]
        data = df_low[cols].dropna()
        exog = sm.add_constant(data[control_vars])
        res = IV2SLS(data[outcome_var], exog, data[[ev]], data[[iv]]).fit()

        if debug:
            print(f"{ev} (low): {res.params[ev]:.4f} ({res.std_errors[ev]:.4f})")

        idx = id_li[7 + i]
        solutions[idx] = Solution(idx, title, query_li[7 + i],
                                  method, dataset_name,
                                  res.params[ev], res.std_errors[ev],
                                  treat_var=ev, outcome_var=outcome_var,
                                  control_vars=control_vars, instrument_var=iv, is_rct=True)

    return solutions


def build_paper43(debug=False):
    """
    Builds the representation of paper 43

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Timing is everything? Primacy and recency effects in voter mobilization campaigns"
    dataset_name = "panagopoulos_timing"
    year = 2011
    domain = "political science"
    n_solutions = 9

    queries = ["What is the effect of being contacted by a get-out-the-vote phone call on voter turnout in the November 2005 Rochester, NY election?",
        "What is the effect of receiving a get-out-the-vote phone call four weeks prior to the election on voter turnout?",
        "What is the effect of receiving a get-out-the-vote phone call two weeks prior to the election on voter turnout?",
        "What is the effect of receiving a get-out-the-vote phone call three days prior to the election on voter turnout?",
        "Among high-propensity voters, defined as those with a voting propensity greater than 0.68, what is the effect of receiving a get-out-the-vote phone call four weeks prior to the election on turnout?",
        "Among high-propensity voters, defined as those with a voting propensity greater than 0.68, what is the effect of receiving a get-out-the-vote phone call two weeks prior to the election on turnout?",
        "Among high-propensity voters, defined as those with a voting propensity greater than 0.68, what is the effect of receiving a get-out-the-vote phone call three days prior to the election on turnout?",
        "Among low-propensity voters, defined as those with a voting propensity of at most 0.68, what is the effect of receiving a get-out-the-vote phone call four weeks prior to the election on turnout?",
        "Among low-propensity voters, defined as those with a voting propensity of at most 0.68, what is the effect of receiving a get-out-the-vote phone call three days prior to the election on turnout?",]

    ids = list(range(72, 81))

    solutions = replicated_paper43(title, dataset_name, queries, ids, debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
