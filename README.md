# A Hierarchical Bayesian Model of Hospital Charges

**Bayesian Inference & Decision Theory — Group Project**

## Team

| Name | GitHub Username |
|------|------------------|
| Harina Chohan | [@HarinaChohan](https://github.com/HarinaChohan) |
| Hetal Khumbarana | [@HetalK4](https://github.com/HetalK4) |
| Marilyn Maika | [@MarilynMaika](https://github.com/MarilynMaika) |
| Selmah Tzindori | [@SelmahT](https://github.com/SelmahT) |
| Queen Esther | [@QUEEN-KIBEGI](https://github.com/QUEEN-KIBEGI) |

## Overview

This project builds a Bayesian hierarchical (partial-pooling) model of hospital charges
using the SPARCS (Statewide Planning and Research Cooperative System) hospital inpatient
discharge dataset for New York State. Patients are nested within hospital facilities, and
facility sample sizes are highly unequal — a structure that makes hierarchical modeling
the natural choice over either ignoring facility identity (complete pooling) or fitting
each facility separately (no pooling).

**Core question:** how much do hospital charges vary by facility, after accounting for
patient and admission characteristics — and is that variation large enough to justify
modeling it explicitly?

## Repository Structure

```text
├── data/
│   ├── raw/                         # Raw dataset
│   └── processed/
│       └── hospital_data_clean.csv  # Cleaned, modeling-ready dataset
│
├── notebooks/                       # Data preprocessing, EDA, and Bayesian model-fitting notebooks
│
├── docs/                            # Project documentation
│   ├── Preprocessing report
│   ├── EDA report
│   ├── Workflow notes
│   └── Topic decision record
│
└── slides/                          # Final presentation deck and supporting figures
```

## Dataset

- **Source:** SPARCS hospital inpatient discharge records (NY)
- **Raw size:** 2,101,588 rows × 33 columns
- **After facility ID cleanup:** 2,090,946 rows, 205 identified facilities
- **After group-size handling:** 201 facilities (facilities with <30 patients dropped;
  facilities with >500 patients randomly capped at 500 to keep computation tractable
  while preserving each facility's relative size)
- **Final modeling dataset:** 94,289 rows × 8 columns

**Outcome:** `charges` (Total Charges) — selected over Total Costs and Length of Stay
after EDA showed it carried the strongest, most interpretable between-facility variation
tied to institutional pricing.

**Grouping variable:** `facility` — 201 hospitals.

**Predictors:**

| Predictor | Type | Description |
|---|---|---|
| `age_group` | categorical | Patient age bracket (0–17 through 70+) |
| `admission_type` | categorical | Emergency, Elective, Newborn, Urgent, Trauma |
| `severity_code` | categorical | Clinical severity level (0–4; rare code 0 merged into 1) |
| `med_surg` | categorical | Medical vs. Surgical admission ("Not Applicable" merged into Medical) |
| `ed_indicator` | binary | Whether the admission came through the ED |
| `length_of_stay` | numeric | Days in hospital ("120+" censored values capped at 120) |

## Preprocessing Pipeline

1. **Load** — 2.1M rows, 33 columns → 8 relevant columns kept for a memory-light load
2. **Facility ID cleanup** — drop rows with redacted/missing facility IDs
3. **Group-size handling** — drop facilities with <30 patients; cap large facilities at
   500 via random sampling
4. **Define outcome** — select Total Charges; log-transform for modeling
   (`log_charges = log(charges)`)
5. **Clean Length of Stay** — recode the "120+" censoring flag to a numeric value of 120
6. **Finalize** — 94,289 rows × 8 columns, ready for modeling

See `docs/` for the detailed preprocessing and EDA report with full statistics and figures.

## Key EDA Findings

- **Charges are heavily right-skewed** (skewness = 15.22 on the raw scale); log-transformed
  charges are approximately normal — the basis for the Gaussian-on-log-scale likelihood
- **Facility-level variation is substantial** — facility median log(charges) spans roughly
  8.5 to 12.5, confirming real, modelable differences between hospitals
- **All categorical predictors show visible relationships with charges** — age, admission
  type, severity, and medical/surgical status all carry signal
- **Length of stay is the strongest single predictor** (r ≈ 0.59 with log-charges in
  earlier exploratory correlation checks)

## Model Specification

**Likelihood** (Gaussian on log-transformed charges):

The response variable is the natural logarithm of hospital charges:

$$
y_i = \log(\text{charges}_i)
$$

The transformed response is assumed to follow a normal distribution:

$$
y_i \sim \text{Normal}(\mu_i, \sigma)
$$

where:

- $y_i$ is the log-transformed hospital charge for patient $i$.
- $\mu_i$ is the expected (mean) log charge predicted by the model.
- $\sigma$ is the residual standard deviation, representing unexplained variability.

### Linear Predictor

The expected log hospital charge for patient $i$ is modeled as:

$$
\mu_i = \beta_0 + \beta_1(\text{ageGroup}_i) + \beta_2(\text{admissionType}_i) + \beta_3(\text{severityCode}_i) + \beta_4(\text{medSurg}_i) + \beta_5(\text{edIndicator}_i) + \beta_6(\text{lengthOfStay}_i) + u_{j[i]}
$$

where:

- $\beta_0$ is the overall (population-level) intercept.
- $\beta_1$–$\beta_6$ are regression coefficients that quantify the effect of each predictor on the log of hospital charges.
- $\text{ageGroup}_i$ represents the patient's age category (`age_group` in the dataset).
- $\text{admissionType}_i$ indicates the type of hospital admission (`admission_type` in the dataset).
- $\text{severityCode}_i$ represents the patient's illness severity level (`severity_code` in the dataset).
- $\text{medSurg}_i$ indicates whether the patient underwent a medical or surgical procedure (`med_surg` in the dataset).
- $\text{edIndicator}_i$ indicates whether the patient was admitted through the emergency department (`ed_indicator` in the dataset).
- $\text{lengthOfStay}_i$ is the patient's hospital stay in days (`length_of_stay` in the dataset).
- $u_{j[i]}$ is the facility-specific random intercept for the hospital where patient $i$ received treatment.

### Facility-Level Random Intercept

To account for differences between hospitals that are not explained by the observed patient characteristics, each facility is assigned its own random intercept:

$$
u_j \sim \text{Normal}(0, \sigma_{\text{facility}}), \qquad j = 1, \ldots, 201
$$

where:

- $u_j$ is the random effect for facility $j$.
- The random effects are assumed to have a mean of **0**, implying that facilities are centered around the overall average hospital charge.
- $\sigma_{\text{facility}}$ is the standard deviation of the facility-level random effects and measures the amount of variation in charges across hospitals after accounting for patient-level predictors.
- The model includes **201 facility-level random intercepts**, one for each hospital in the dataset, allowing each facility to have its own baseline level of hospital charges while sharing information through the hierarchical Bayesian framework.

An **identity link** is used throughout — predictors act directly on the expected log-charge.

### Priors

Weakly informative, data-scaled priors (Bambi's automatic defaults):

| Parameter | Prior | Purpose |
|---|---|---|
| Intercept | Normal(μ=10.28, σ=7.75) | Centered at ~$29,000; wide, allows a broad baseline range |
| Fixed effects (β) | Normal(μ=0, σ=5.7–64.2 depending on predictor) | Regularizes without dominating the likelihood; wider for rarer categories |
| Facility σ | HalfNormal(σ=7.75) | Allows substantial between-facility variation, constrained positive |
| Residual σ | HalfStudentT(ν=4, σ=1.11) | Heavy-tailed, robust to outliers |

## Estimation

- **Method:** Hamiltonian Monte Carlo via the No-U-Turn Sampler (NUTS)
- **Backend:** Numpyro (GPU-accelerated where available)
- **Final configuration:** 4 chains, 2,000 warmup iterations, 2,000 posterior draws
  (8,000 total samples)
- **Log-likelihood** stored at fit time (`idata_kwargs={"log_likelihood": True}`) to
  enable WAIC/LOO model comparison

Convergence was tested at 1,000, 2,000, and 4,000 iterations, plus a non-centered
reparameterization. The 2,000-iteration fit was selected as final — best convergence for
the Intercept (R-hat 1.03) with acceptable convergence overall (R-hat ≤ 1.03 across all
parameters), at half the runtime of the 4,000-iteration alternative.

## Results

**Posterior estimates (final model):**

| Parameter | Mean | 94% HDI | Interpretation |
|---|---|---|---|
| Intercept | 9.56 | [9.45, 9.67] | Baseline log(charges) |
| length_of_stay | 0.051 | [0.051, 0.052] | +1 day → ~5.2% higher charges |
| facility σ | 0.627 | [0.553, 0.691] | Facilities vary by ~±1.23 on the log scale |
| residual σ | 0.506 | [0.504, 0.508] | Unexplained, patient-level variation |

**Model comparison (hierarchical vs. a flat baseline with no facility effect), via WAIC:**

| Model | elpd_waic | Weight |
|---|---|---|
| Hierarchical | −69,695.5 | 0.978 |
| Baseline (no facility) | −109,539.0 | 0.022 |

elpd_diff ≈ 39,844 — decisive evidence that facility effects carry real predictive value.

**Facility shrinkage:** 5 of 201 facilities are significantly different from the
population average (2 higher-charging, 3 lower-charging); the rest show the expected
shrinkage pattern, clustering near the global mean with wider uncertainty for
smaller facilities.

**Posterior predictive checks:** the baseline model predicts the mean charge slightly
better (bias 0.194 vs. 0.340), but the hierarchical model captures the true spread in
charges more accurately (std-dev bias 0.179 vs. 0.195) — consistent with its purpose of
distributing variance across facilities rather than treating it all as noise.

## Limitations

- **Facility sampling cap** — large facilities were randomly capped at 500 records,
  which doesn't reflect true facility volume and could understate variance for
  high-volume hospitals
- **Excluded predictors** — APR MDC Description, Patient Disposition, and Risk of
  Mortality were dropped for parsimony or redundancy; some residual confounding may
  remain in the facility effect
- **Imperfect convergence** — R-hat stayed at or below 1.03 for the Intercept and
  facility σ across every iteration count tried, never fully reaching the standard
  <1.01 target
- **Anonymized facility identity** — facility IDs carry no metadata (teaching status,
  urban/rural, bed count), so the model can measure that facilities differ without
  explaining why
- **Homogeneous residual variance** — a single σ is assumed across all facilities and
  severity levels
- **Association, not causation** — the facility effect should be read as a pricing
  association, not a causal claim

## Status

- [x] Topic chosen: Hierarchical Bayesian Models
- [x] Dataset identified and cleaned
- [x] EDA complete
- [x] Priors specified and validated
- [x] Model fitted, converged (acceptable), and compared against a baseline
- [x] Posterior predictive checks complete
- [x] Slide deck prepared (`slides/`)
- [ ] Final write-up / report

## Reproducing This Project

```bash
pip install -U numpy pandas matplotlib seaborn scipy pymc bambi arviz numpyro tqdm
```

Run notebooks in `notebooks/` in order: preprocessing → EDA → model fitting. The cleaned
dataset (`hospital_data_clean.csv`) is produced by the preprocessing notebook and consumed
by every notebook after it.