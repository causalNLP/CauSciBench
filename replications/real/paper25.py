## This file contains the replication of paper 25
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 25

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper25(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 25
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "ever_treated"
    outcome_var = "Med_per_inmateday"
    time_var = "year"
    method = "did"

    did_data = df[df['In_Reuters_Data'] == 1].copy()
    did_data['post_treatment'] = ((did_data['year'] >= did_data['first.treated']) & (did_data['first.treated'] != 0)).astype(int)
    did_data['ever_treated'] = (did_data['first.treated'] != 0).astype(int)

    ever = did_data[did_data['ever_treated'] == 1]
    model = smf.ols(f"{outcome_var} ~ post_treatment", data=ever).fit(cov_type="HC1")

    if debug:
        print(model.summary())

    answer = model.params["post_treatment"]
    std_err = model.bse["post_treatment"]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         time_var=time_var, canonical_did=True, is_rct=False)

    ## Solution 2: TWFE DiD effect of privatization on overall inmate death rate
    ds2 = "zoorob_privatization"
    df2_raw = pd.read_csv(BASE_DIR / f"{ds2}.csv", encoding="latin-1", engine="python")
    df2 = df2_raw.dropna(subset=['death_rate', 'private_provider']).copy()

    m2 = smf.ols('death_rate ~ private_provider + C(id) + C(year)', data=df2).fit(cov_type='HC1')
    if debug:
        print(m2.summary())

    solution2 = Solution(id_li[1], title, query_li[1], method, ds2,
                         float(m2.params['private_provider']), float(m2.bse['private_provider']),
                         treat_var='private_provider', outcome_var='death_rate',
                         state_var='id', time_var='year', is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper25(debug=False):
    """
    Builds the representation of paper 25

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Privatization and quality of carceral healthcare: A difference-in-differences analysis of jails in the united states, 2008-2019"
    dataset_name = "zoorob_privatization_va"
    year = 2024
    domain = "health economics"
    n_solutions = 2

    query1 = "What is the effect of privatizing jail healthcare on medical spending per inmate-day in Virginia?"
    query2 = "What is the effect of privatizing jail healthcare on overall inmate death rates?"

    solutions = replicated_paper25(title, dataset_name, [query1, query2], [46, 154], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
