# Data

This folder contains the datasets used in CauSciBench along with their corresponding metadata. The benchmark covers three data splits: QRData (39 queries), real-world replications (175 queries), and synthetic data (143 queries).

The CSV files can be accessed from Hugging Face. Please download the `csv_files` folder and add it here. For now, `data/csv_files` is empty and is meant to be a placeholder.

```
data/
├── csv_files/
│   ├── qrdata/             # CSV files for the QRData split (35 datasets)
│   ├── realdata/           # CSV files for the real-world split (101 datasets)
│   └── synthetic_data/     # CSV files for the synthetic split (143 datasets)
├── metadata_json/          # Benchmark annotations in JSON format
│   ├── qr_input.json       # QRData split (39 queries)
│   ├── real_input.json     # Real-world split (175 queries)
│   └── synthetic_input.json  # Synthetic split (143 queries)
└── metadata_csv/           # Benchmark annotations in CSV format
    ├── qr_input.csv        # QRData split (39 queries)
    ├── real_input.csv      # Real-world split (175 queries)
    └── synthetic_input.csv # Synthetic split (143 queries)
```

## Annotation Details

Each record is annotated with the following fields pertaining to causal inference:

1. Dataset description — variable definitions, data source, and collection mechanism
2. Causal query — the natural language query that can be answered from the associated dataset
3. CSV filename — CSV file associated with the query
4. Reference causal method — the method used to produce the reference answer (e.g., OLS, IV, RDD, DiD, etc.)
5. Causal effect estimate
6. Standard error
7. Treatment variable
8. Outcome variable
9. Control variables / observed confounders
10. Whether the data comes from a randomized trial (`is_rct`)
11. Method-specific variables:
    - `instrument_var` — for instrumental variable (IV) designs
    - `running_var` — for regression discontinuity (RDD) designs
    - `temporal_var` — time indicator for difference-in-differences (DiD)
    - `state_var` — entity identifier for panel data / TWFE version of DiD
    - `mediator` — mediating variable for frontdoor identification (synthetic split only)

## Metadata CSV Columns

| Column | Description |
|---|---|
| `id` | Unique query identifier |
| `paper_name` | Source paper |
| `data_description` | Natural language description of the dataset and variables |
| `dataset_name` | CSV filename within the relevant `csv_files/` subdirectory |
| `query` | The causal question |
| `method` | Estimation method (`ols`, `iv`, `rdd`, `did`, `matching`, `frontdoor`, `ipw`, `glm`) |
| `answer` | Reference causal effect estimate |
| `std_error` | Standard error of the estimate |
| `treatment` | Name of the treatment variable |
| `outcome` | Name of the outcome variable |
| `controls` | Comma-separated list of control variables |
| `instrument_var` | Instrumental variable|
| `running_var` | Running variable |
| `temporal_var` | Time / post-treatment indicator |
| `state_var` | Entity / panel identifier |
| `interaction_var` | Interaction variable (variable interacting with the treatment) |
| `is_rct` | 1 if from a randomized controlled trial, 0 otherwise |
| `is_multirct` | 1 if there are multiple treatment arms, 0 otherwise |

## License

### 1. Real Data

#### CC0 1.0

| Dataset | Source |
|---|---|
| gerber_social_pressure.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/CGMWNW |
| vernby_can_immigrants.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HVRL0S |
| thomas_effects_of.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CIPVBK |
| tan_causal_effect.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KCSVWH |
| liu_public_trust1.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ATDFQN |
| liu_public_trust2.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ATDFQN |
| angrist_does_compulsory.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ENLGZX |
| angrist_using_maimonides.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XRSUJU |
| gurun_do_wall_street.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HCWJRW |
| zoorob_privatization_va.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/91S9AP |
| leigh_estimating_the_impact.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JPDQ5P |
| broockman_do_congressional.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/IOEQKE |
| venkatramani_early_medicaid.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/9ZS4KR |
| eckhouse_metrics_management.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3E7JXB |
| lusher_double_shift.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6AYLVO |
| sabia_are_minimum.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6AYLVO |
| barton_understanding.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KULZYU |
| carpenter_the_minimum_legal.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Q9VQIU |
| thompson_how_partisan.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CFASH6 |
| dhingra_immigration_policies.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/RGZWNJ |
| xiong_effect_of.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HHSREH |
| krasno_do_televised.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/QJKB0I |
| alvarez_voting_made_easy.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/24896 |
| butler_can_learning.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/QVCG0C |
| goldsmith_doing_well.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IHMFPJ |
| butler_were_newspapers.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/OAYRDW |
| malesky_foreign.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDVSW5 |
| panagopoulos_timing.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/STKD55 |
| oswald_computational.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/G37BHE |
| rickard_incumbents.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/X1JL2H |
| xu_does_mislabelling.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CDUROD |
| urdinez_undermining.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KNG7CY |
| urdinez_china.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EIAXSE |
| calonico_regression.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LPZLBF |
| dorussen_the_influence.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AQYYNK |
| trein_europeanisation.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0T5AN0 |
| zoorob_does_right.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UVCZ5D |
| alfaro_the_effect.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YZOUI2 |
| malesky_impact_of_recentralization.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IUG2C4 |
| tan_impact_of_intergenerational.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YE9B6 |
| song_estimating_incumbency.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JSOWUR |
| angrist_vouchers.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K57TOZ |
| abadie_instrumental_variables.csv | https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/XVADZ7/OYIAFS |
| panagopolous_field_experiments.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/NTMEZU |
| uji_comparing_public.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HM1KXT |
| rebeira_does_rising.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/E3X2NO |
| conley_inference_with_merit.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GCBK24 |
| conley_inference_with_hope.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GCBK24 |
| vertier_dismantling.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/RQFXPR |
| dwiputri_the_corruption.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8VF8RV |
| gonzalez_blame_shifting.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QNCYOQ |
| white_evaluating_the_minority.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/U5PDBR |
| taddeo_causal.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ENG0IY |
| newman_global_costs.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N3ED1N |
| garcia_colored_perceptions.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QTTA73 |
| hawes_social_capital.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B8NR76 |
| hawes_give_us.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L1V6BP |
| lee_impact.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XZJZAN |
| soules_call_to.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6BZSKR |
| cai_effect_microinsurance.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/27174 |
| pereira_does_electing.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NSALSE |
| keele_do_term_limits.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/23123 |
| deuchert_direct.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/I57L3O |
| bokemper_experimental_evidence.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NZYWS5 |
| goldstein_lobbyists_recovery.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/RSD5BV |
| goldstein_lobbyists_earmark.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JTSYV3 |
| broockman_do_female.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/23624 |
| snyder_partisan_imbalance.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/23624 |
| abouchadi_causal_effect.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/23624 |
| reinsberg_does_earmarked.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/23624 |
| cirone_cabinets.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F1VLBI |
| chi_improving.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TBBV8R |
| zoorob_privatization.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/91S9AP |
| foos_all_in.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZFLG25 |
| butler_do_politicians.csv | https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/VAHSO4 |
| sharkey_community.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/46WIH0 |
| brogaard_interpreting_performance1.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JLHLJG |
| brogaard_interpreting_performance2.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JLHLJG |
| carpenter_the_minimum_legal_drinking.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/27070 |
| bowles_countering.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MDF4SO |

#### MIT License

| Dataset | Source |
|---|---|
| card_using_geographic.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| cunningham_the_long_run.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| broockman_black_politicians.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| cheng_does_strengthening.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| manacorda_gov_transfers.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| kessler_dont_take.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| thornton_the_demand.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| lee_do_voters.csv | https://cran.r-project.org/web/packages/causaldata/index.html |
| graddy_markets_fulton.csv | https://github.com/scunning1975/mixtape |
| hansen_punishment.csv | https://github.com/scunning1975/mixtape |
| hartford_deepiv.csv | https://github.com/jhartford/DeepIV |

#### CC BY 4.0

| Dataset | Source |
|---|---|
| autor_china_syndrome.csv | https://www.openicpsr.org/openicpsr/project/112670/version/V1/view |
| wilson_internet.csv | https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002479 |
| enache_demand_for.csv | https://www.openicpsr.org/openicpsr/project/186201/version/V1/view |
| ruth_partisan_conflict.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UM4VAJ |

#### CC BY-NC 2.0

| Dataset | Source |
|---|---|
| dehejia_propensity_score.csv | https://users.nber.org/~rdehejia/data/.nswdata2.html |
| dehejia_causal_effects.csv | https://users.nber.org/~rdehejia/data/.nswdata2.html |
| lalonde_evaluating.csv | https://users.nber.org/~rdehejia/data/.nswdata2.html |

#### GPL 3.0

| Dataset | Source |
|---|---|
| lee_randomized_experiments.csv | https://search.r-project.org/CRAN/refmans/RATest/html/lee2008.html |

#### No License Specified

| Dataset | Source |
|---|---|
| card_minimum_wages.csv | https://davidcard.berkeley.edu/data_sets.html |
| ho_matching.csv | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/RWUY8G |

### 2. QRData

#### MIT License

MPs.csv, ak91.csv, app_engagement_push.csv, billboard_impact.csv, collections_email.csv, drinking.csv, hospital_treatment.csv, ihdp_0.csv, ihdp_1.csv, ihdp_2.csv, ihdp_3.csv, ihdp_4.csv, ihdp_5.csv, ihdp_6.csv, ihdp_7.csv, ihdp_8.csv, ihdp_9.csv, jobs_0.csv, jobs_1.csv, jobs_2.csv, jobs_3.csv, jobs_4.csv, jobs_5.csv, jobs_6.csv, jobs_7.csv, jobs_8.csv, jobs_9.csv, learning_mindset.csv, medicine_impact_recovery.csv, online_classroom.csv, smoking2.csv, social.csv, trainee_unique_on_age.csv, wage.csv, women.csv

All files can be obtained from: https://github.com/xxxiaol/QRData

### 3. Synthetic Data

#### MIT License

The synthetic data was generated by the authors of this benchmark. All files in `csv_files/synthetic_data/` are released under the MIT License.
