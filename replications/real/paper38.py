## This file contains the replication of paper 38
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from solution import Solution, Paper

PAPER_ID = 38

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper38(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 38
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "EV"
    control_vars = ["age.group", "educ", "tech", "pol.info", "white.collar", "not.full.time", "male"]
    method = "matching"

    outcome_vars = ["eselect.cand", "eval.voting", "easy.voting", "agree.evoting",
        "how.clean", "sure.counted", "capable.auth", "speed", "conf.secret"]

    df.iloc[:, 9:18] = df.iloc[:, 9:18].fillna(99999)
    df = df.dropna()
    df.replace(99999, np.nan, inplace=True)
    df.iloc[:, 9:18] = df.iloc[:, 9:18].fillna(99999)

    for col in ["age.group", "educ", "tech", "pol.info"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = pd.DataFrame({
        "age":        df["age.group"],
        "age2":       df["age.group"] ** 2,
        "age3":       df["age.group"] ** 3,
        "educ":       df["educ"],
        "educ2":      df["educ"] ** 2,
        "tech":       df["tech"],
        "tech2":      df["tech"] ** 2,
        "pol_info":   df["pol.info"],
        "age_x_educ": df["age.group"] * df["educ"],
        "age_x_tech": df["age.group"] * df["tech"],
        "educ_x_pol": df["educ"] * df["pol.info"],
        "age_x_pol":  df["age.group"] * df["pol.info"],
        "tech_x_pol": df["tech"] * df["pol.info"],
        "white.collar":  pd.to_numeric(df["white.collar"],  errors="coerce").fillna(0),
        "not.full.time": pd.to_numeric(df["not.full.time"], errors="coerce").fillna(0),
        "male":          pd.to_numeric(df["male"],           errors="coerce").fillna(0),
    }, index=df.index)

    np.random.seed(36466)
    ps_rows = X.dropna().index
    X_fit = X.loc[ps_rows]
    y_fit = df.loc[ps_rows, treat_var].astype(int)

    lr = LogisticRegression(solver="liblinear", max_iter=1000).fit(X_fit, y_fit)
    ps = pd.Series(lr.predict_proba(X_fit)[:, 1], index=ps_rows, name="propensity_score")
    df = df.copy()
    df["propensity_score"] = ps

    caliper = 0.05
    treated_idx = df.query("EV==1").index.intersection(ps_rows)
    control_idx  = df.query("EV==0").index.intersection(ps_rows)
    treated_ps   = df.loc[treated_idx, "propensity_score"]
    control_ps   = df.loc[control_idx, "propensity_score"]

    nn = NearestNeighbors(n_neighbors=1).fit(control_ps.values.reshape(-1, 1))
    matched_treated, matched_controls, control_used = [], [], set()
    for t_idx in treated_ps.sort_values().index:
        t_ps = treated_ps[t_idx]
        _, pos = nn.kneighbors([[t_ps]])
        c_idx = control_ps.index[int(pos[0][0])]
        if abs(t_ps - control_ps[c_idx]) <= caliper and c_idx not in control_used:
            matched_treated.append(t_idx); matched_controls.append(c_idx); control_used.add(c_idx)
        else:
            diffs = (control_ps - t_ps).abs()[~control_ps.index.isin(control_used)]
            if not diffs.empty and diffs.min() <= caliper:
                nearest = diffs.idxmin()
                matched_treated.append(t_idx); matched_controls.append(nearest); control_used.add(nearest)

    datamatched = df.loc[matched_treated + matched_controls].replace(99999, np.nan)

    if debug:
        print(f"Matched pairs: {len(matched_treated)}")

    solutions = {}
    for i, outcome_var in enumerate(outcome_vars):
        y = datamatched[outcome_var]
        ev_vals = y[datamatched[treat_var] == 1].dropna()
        tv_vals = y[datamatched[treat_var] == 0].dropna()

        p_ev = ev_vals.mean()
        p_tv = tv_vals.mean()
        diff = (p_ev - p_tv) * 100
        se   = np.sqrt(p_ev * (1 - p_ev) / len(ev_vals) + p_tv * (1 - p_tv) / len(tv_vals)) * 100

        if debug:
            print(f"{outcome_var}: diff={diff:.4f}, se={se:.4f}")

        solution = Solution(id_li[i], title, query_li[i], method, dataset_name,
                            diff, se, treat_var=treat_var, outcome_var=outcome_var,
                            control_vars=control_vars, is_rct=False)
        solutions[id_li[i]] = solution

    return solutions


def build_paper38(debug=False):
    """
    Builds the representation of paper 38

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Voting made safe and easy: The impact of e-voting on citizen perceptions"
    dataset_name = "alvarez_voting_made_easy"
    year = 2013
    domain = "political science"
    n_solutions = 9

    queries = ["Does electronic voting make it easier or harder for people to vote compared to paper ballots?",
        "Does using e-voting improve voters' overall evaluation of the voting experience?",
        "How does being assigned to e-voting instead of traditional voting affect the perceived ease of the voting procedure?",
        "What is the effect of e-voting on voters' agreement to replace traditional voting with electronic voting?",
        "Are voters more likely to trust election integrity when they vote electronically vs. on paper?",
        "Does electronic voting make voters more or less confident their vote was counted compared to paper voting?",
        "What is the effect of e-voting on voters' perception of poll worker qualification compared to traditional voting?",
        "Does e-voting change voters' perception of the speed of the voting process compared to traditional voting?",
        "What is the effect of e-voting compared to traditional voting on voters' confidence in ballot secrecy?",]

    ids  = list(range(59, 68))

    solutions = replicated_paper38(title, dataset_name, queries, ids, debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
