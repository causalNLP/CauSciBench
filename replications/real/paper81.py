## This file contains the replication of paper 81
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from solution import Solution, Paper

PAPER_ID = 81

BASE_DIR = Path("../data/csv_files/realdata/")


def _effects(y, z, d, t):
    p11 = np.mean(d[z == 1])
    p10 = np.mean(d[z == 0])
    p01 = 1 - p11
    comp = p11 - p10
    LATE = (np.mean(y[(t == 1) & (z == 1)]) - np.mean(y[(t == 1) & (z == 0)])) / comp
    comp0 = (np.mean(y[(t == 1) & (z == 0) & (d == 0)]) -
             p01 / comp * (np.mean(y[(t == 0) & (z == 1) & (d == 0)]) -
                          np.mean(y[(t == 0) & (z == 0) & (d == 0)])))
    comp1 = (np.mean(y[(t == 1) & (z == 1)] * d[(t == 1) & (z == 1)]) -
             np.mean(y[(t == 1) & (z == 0)] * d[(t == 1) & (z == 0)])) / comp
    totalcomp = comp1 - comp0
    ate = np.mean(y[(t == 1) & (z == 1)]) - np.mean(y[(t == 1) & (z == 0)])
    dir0 = (np.mean(y[(t == 1) & (z == 1) & (d == 0)]) -
            np.mean(y[(t == 0) & (z == 1) & (d == 0)]) -
            (np.mean(y[(t == 1) & (z == 0) & (d == 0)]) -
             np.mean(y[(t == 0) & (z == 0) & (d == 0)])))
    indir1 = totalcomp - dir0
    return np.array([ate, dir0, totalcomp, dir0, indir1, LATE, indir1 - LATE])


def _bootstrap(y, z, d, t, boot, clusters, key):
    mc = []
    for _ in range(boot):
        sboot = np.random.choice(clusters, size=len(clusters), replace=True)
        db, yb, zb, tb = [], [], [], []
        for k in sboot:
            mask = key == k
            db.extend(d[mask]); zb.extend(z[mask])
            yb.extend(y[mask]); tb.extend(t[mask])
        mc.append(_effects(np.array(yb), np.array(zb), np.array(db), np.array(tb)))
    return np.array(mc)


_qty_names = ['ate', 'd_nt', 't_co_A6', 'd_co_0', 'in_co_1_A6', 'LATE', 'in_co_1_A6_minus_LATE']


def _run(outcome, data, boot=1000):
    ind = ((~np.isnan(data[outcome])) & (~np.isnan(data['treatment1'])) &
           (~np.isnan(data['ceiling'])) & (data['wave'] >= 4))
    y = data.loc[ind, outcome].values
    z = data.loc[ind, 'ceiling'].values
    d = data.loc[ind, 'treatment1'].values
    t = data.loc[ind, 'after'].values
    key = data.loc[ind, 'id'].values
    clusters = np.unique(key)
    eff = _effects(y, z, d, t)
    mc = _bootstrap(y, z, d, t, boot=boot, clusters=clusters, key=key)
    sd = np.std(mc, axis=0)
    qty = dict(zip(_qty_names, zip(eff, sd)))
    return qty


def replicated_paper81(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 81
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df['after'] = df['wave'] > 4

    treat_var = 'ceiling'
    time_var = 'wave'
    method = 'did'
    np.random.seed(42)

    ## Solution 1: d_nt on polalient (direct effect on never-takers, skeptical gov attitudes)
    r1 = _run('polalient', df)
    answer1, std_err1 = r1['d_nt']
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var='polalient',
                         state_var='id', time_var=time_var, is_rct=False)

    ## Solution 2: in_co_1_A6 on demo (indirect effect on compliers, mild democrat)
    r2 = _run('demo', df)
    answer2, std_err2 = r2['in_co_1_A6']
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treat_var, outcome_var='demo',
                         state_var='id', time_var=time_var, is_rct=False)

    ## Solution 3: d_co_0 on srepu (direct effect on compliers, strong republican)
    r3 = _run('srepu', df)
    answer3, std_err3 = r3['d_co_0']
    solution3 = Solution(id_li[2], title, query_li[2], method, dataset_name,
                         answer3, std_err3, treat_var=treat_var, outcome_var='srepu',
                         state_var='id', time_var=time_var, is_rct=False)

    ## Solution 4: d_nt on repu (direct effect on never-takers, mild republican)
    r4 = _run('repu', df)
    answer4, std_err4 = r4['d_nt']
    solution4 = Solution(id_li[3], title, query_li[3], method, dataset_name,
                         answer4, std_err4, treat_var=treat_var, outcome_var='repu',
                         state_var='id', time_var=time_var, is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2,
            id_li[2]: solution3, id_li[3]: solution4}


def build_paper81(debug=False):
    """
    Builds the representation of paper 81

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Direct and indirect effects based on difference-in-differences with an application to political preferences following the Vietnam draft lottery"
    dataset_name = "deuchert_direct"
    year = 2019
    domain = "political science"
    n_solutions = 4

    query1 = "What is the effect of being drafted on those who never serve in the military independently of the draft lottery, not explained by serving in the military, with respect to having positive attitudes towards the government?"
    query2 = "What is the effect of being drafted on the compliers among those who comply with the draft lottery, explained by serving in the military, with respect to holding mild democrat views?"
    query3 = "What is the effect of being drafted on the compliers among those who comply with the draft lottery, not explained by serving in the military, with respect to holding strong republican views?"
    query4 = "What is the effect of being drafted among those who never serve in the military independently of the draft lottery, not explained by serving in the military, with respect to holding mild republican views?"

    solutions = replicated_paper81(title, dataset_name,
                                   [query1, query2, query3, query4],
                                   [129, 130, 131, 132], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
