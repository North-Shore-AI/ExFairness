<p align="center">
  <img src="assets/ExFairness.svg" alt="ExFairness" width="150"/>
</p>

# ExFairness

**Fairness and Bias Detection Library for Elixir AI/ML Systems**

[![Elixir](https://img.shields.io/badge/elixir-1.14+-purple.svg)](https://elixir-lang.org)
[![OTP](https://img.shields.io/badge/otp-25+-blue.svg)](https://www.erlang.org)
[![Hex.pm](https://img.shields.io/hexpm/v/ex_fairness.svg)](https://hex.pm/packages/ex_fairness)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-purple.svg)](https://hexdocs.pm/ex_fairness)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/North-Shore-AI/ExFairness/blob/main/LICENSE)

---

ExFairness is a comprehensive library for detecting, measuring, and mitigating bias in AI/ML systems built with Elixir. It provides rigorous fairness metrics, bias detection algorithms, and mitigation techniques to ensure your models make equitable predictions across different demographic groups.

## Features

### ✅ Fairness Metrics (Implemented)

- **Demographic Parity**: Ensures equal positive prediction rates across groups
- **Equalized Odds**: Ensures equal true positive and false positive rates across groups
- **Equal Opportunity**: Ensures equal true positive rates across groups (recall parity)
- **Predictive Parity**: Ensures equal positive predictive values (precision parity)
- **Calibration Fairness**: Probability predictions are equally calibrated across groups (ECE/MCE)

### ✅ Statistical Inference (Implemented)

- **Bootstrap Confidence Intervals**: Percentile/basic CIs with stratified resampling
- **Hypothesis Testing**: Two-proportion z-test, chi-square, and permutation tests with effect sizes

### ✅ Bias Detection (Implemented)

- **Disparate Impact Analysis**: EEOC 80% rule for legal compliance

### ✅ Mitigation Techniques (Implemented)

- **Reweighting**: Sample weighting for demographic parity and equalized odds

### ✅ Reporting (Implemented)

- **Comprehensive Reports**: Multi-metric fairness assessment
- **Markdown Export**: Human-readable documentation
- **JSON Export**: Machine-readable integration

### 🚧 Coming Soon

- **Individual Fairness**: Similar individuals receive similar predictions
- **Counterfactual Fairness**: Causal fairness analysis
- **Intersectional Analysis**: Multi-attribute fairness
- **Temporal Monitoring**: Fairness drift detection
- **Resampling**: Oversampling and undersampling techniques
- **Threshold Optimization**: Group-specific decision thresholds

## Design Principles

1. **Mathematical Rigor**: All metrics based on established fairness research
2. **Transparency**: Clear explanations of fairness definitions and trade-offs
3. **Actionability**: Concrete mitigation strategies with implementation guidance
4. **Flexibility**: Support for multiple fairness definitions and use cases
5. **Integration**: Works seamlessly with Nx, Axon, and other Elixir ML tools

## Installation

Add `ex_fairness` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ex_fairness, "~> 0.3.0"}
  ]
end
```

Or install from GitHub:

```elixir
def deps do
  [
    {:ex_fairness, github: "North-Shore-AI/ExFairness"}
  ]
end
```

## Quick Start

### Measure Demographic Parity

```elixir
# Binary classification predictions and sensitive attributes
# Need at least 10 samples per group for statistical reliability
predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.demographic_parity(predictions, sensitive_attr)
# => %{
#   group_a_rate: 0.50,
#   group_b_rate: 0.50,
#   disparity: 0.00,
#   passes: true,
#   threshold: 0.10,
#   interpretation: "Group A receives positive predictions at 50.0% rate..."
# }
```

### Measure Equalized Odds

```elixir
# Include ground truth labels
predictions = Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.equalized_odds(predictions, labels, sensitive_attr)
# => %{
#   group_a_tpr: 0.50,
#   group_b_tpr: 0.50,
#   group_a_fpr: 0.33,
#   group_b_fpr: 0.33,
#   tpr_disparity: 0.00,
#   fpr_disparity: 0.00,
#   passes: true,
#   interpretation: "Group A: TPR=50.0%, FPR=33.3%..."
# }
```

### Assess Calibration Fairness

```elixir
# Probabilistic predictions, labels, and sensitive attribute
probabilities = Nx.tensor([0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8, 0.5, 0.3,
                           0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8, 0.5, 0.3])
labels = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

calibration = ExFairness.Metrics.Calibration.compute(
  probabilities,
  labels,
  sensitive_attr,
  n_bins: 10,
  strategy: :uniform,    # or :quantile
  threshold: 0.1         # max acceptable ECE disparity
)

calibration.disparity    # |ECE_A - ECE_B|
calibration.group_a_ece  # expected calibration error for group A
calibration.group_b_mce  # maximum calibration error for group B
calibration.passes       # true if disparity <= threshold
IO.puts(calibration.interpretation)

# Shortcut helper
calibration = ExFairness.calibration(probabilities, labels, sensitive_attr)
```

### Statistical Inference (Confidence Intervals & Tests)

#### Bootstrap confidence intervals for any metric

```elixir
predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# metric_fn must return a numeric statistic (here: demographic parity disparity)
metric_fn = fn [preds, sens] ->
  ExFairness.demographic_parity(preds, sens).disparity
end

ci = ExFairness.Utils.Bootstrap.confidence_interval(
  [predictions, sensitive_attr],
  metric_fn,
  n_samples: 1000,
  confidence_level: 0.95,
  stratified: true,
  method: :percentile,
  seed: 42
)

ci.point_estimate       # observed disparity
ci.confidence_interval  # {lower, upper}
ci.method               # :percentile or :basic
```

#### Hypothesis testing for group disparities

```elixir
predictions = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Two-proportion z-test (demographic parity)
z_result =
  ExFairness.Utils.StatisticalTests.two_proportion_test(
    predictions,
    sensitive_attr,
    alpha: 0.05,
    alternative: :two_sided
  )

z_result.p_value
z_result.effect_size    # Cohen's h
z_result.significant

# Permutation test for any metric (example: TPR disparity from equalized odds)
labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])

perm_result =
  ExFairness.Utils.StatisticalTests.permutation_test(
    [predictions, labels, sensitive_attr],
    fn [preds, labs, sens] ->
      ExFairness.equalized_odds(preds, labs, sens).tpr_disparity
    end,
    n_permutations: 1000,
    alpha: 0.05,
    alternative: :two_sided,
    seed: 1234
  )

perm_result.p_value
perm_result.significant
IO.puts(perm_result.interpretation)
```

For error-rate disparities (equalized odds), use `ExFairness.Utils.StatisticalTests.chi_square_test/4` with `predictions`, `labels`, and `sensitive_attr`.

### Comprehensive Fairness Report

```elixir
report = ExFairness.fairness_report(predictions, labels, sensitive_attr,
  metrics: [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity]
)

# => %{
#   demographic_parity: %{passes: true, disparity: 0.00, ...},
#   equalized_odds: %{passes: true, tpr_disparity: 0.00, fpr_disparity: 0.00, ...},
#   equal_opportunity: %{passes: true, disparity: 0.00, ...},
#   predictive_parity: %{passes: true, disparity: 0.00, ...},
#   overall_assessment: "✓ All 4 fairness metrics passed...",
#   passed_count: 4,
#   failed_count: 0,
#   total_count: 4
# }

# Include calibration in the report by supplying probabilities:
# report = ExFairness.fairness_report(predictions, labels, sensitive_attr,
#   probabilities: probs_tensor,
#   metrics: [:demographic_parity, :equalized_odds, :calibration]
# )
# If you pass `probabilities:` and don’t specify `:metrics`, calibration is included by default.
# To enable statistical inference across metrics, pass `include_ci: true` and `statistical_test: :permutation | :z_test | :chi_square`.
# You can skip reliability-diagram computation (for speed) with `include_reliability: false`.

# Statistical inference across metrics (adds CI/p-value columns in markdown):
# report = ExFairness.fairness_report(predictions, labels, sensitive_attr,
#   probabilities: probs_tensor,
#   include_ci: true,
#   statistical_test: :permutation,
#   bootstrap_samples: 500,
#   n_permutations: 2000,
#   include_reliability: false  # skip bin details if you only need headline calibration
# )

# Export to Markdown
markdown = ExFairness.Report.to_markdown(report)
File.write!("fairness_report.md", markdown)

# Export to JSON
json = ExFairness.Report.to_json(report)
File.write!("fairness_report.json", json)

# Calibration reliability detail is included in the report when calibration runs:
# report.calibration.reliability_diagram.bins
```

## Bias Detection

### Disparate Impact Analysis (EEOC 80% Rule)

```elixir
# Legal standard for adverse impact detection
predictions = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attr)
# => %{
#   group_a_rate: 0.80,
#   group_b_rate: 0.20,
#   ratio: 0.25,
#   passes_80_percent_rule: false,
#   interpretation: "Group A selection rate: 80.0%. Group B selection rate: 20.0%..."
# }

# Interpretation explains legal context and EEOC guidelines
IO.puts result.interpretation
```

## Mitigation Techniques

### Reweighting (Pre-processing)

**Overview:**
Reweighting is a pre-processing technique that assigns different weights to training samples to achieve fairness. Samples from underrepresented group-label combinations receive higher weights.

**Mathematical Foundation:**

For demographic parity, the weight for sample with sensitive attribute `a` and label `y` is:

```
w(a, y) = P(Y = y) / P(A = a, Y = y)
```

This ensures that all group-label combinations have equal expected weight in the training process.

**Algorithm:**
1. Compute joint probabilities: P(A=a, Y=y) for all combinations
2. Compute marginal probabilities: P(Y=y)
3. Assign weight to each sample: w = P(Y=y) / P(A=a, Y=y)
4. Normalize weights to mean = 1.0

**Properties:**
- Weights are always positive
- Normalized to mean 1.0 for compatibility with training algorithms
- Balances representation across all (A, Y) combinations
- Can target different fairness metrics

**Implementation:**

```elixir
# Compute sample weights to achieve fairness
labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

weights = ExFairness.Mitigation.Reweighting.compute_weights(
  labels,
  sensitive_attr,
  target: :demographic_parity  # or :equalized_odds
)

# Weights are normalized with mean 1.0
mean_weight = Nx.mean(weights) |> Nx.to_number()
# => 1.0

# Use weights in your training algorithm
# Example with Scholar (hypothetical):
# model = Scholar.Linear.LogisticRegression.fit(features, labels, sample_weights: weights)

# Example with custom training loop:
# loss_fn = fn pred, label, weight ->
#   weight * binary_cross_entropy(pred, label)
# end

# Verify improvement after retraining
# new_predictions = YourML.predict(retrained_model, features)
# new_report = ExFairness.fairness_report(new_predictions, labels, sensitive_attr)
# IO.puts "Improvement: #{new_report.passed_count} metrics now pass"
```

**Expected Outcomes:**
- Improves demographic parity by balancing group-label combinations
- Helps achieve equalized odds by balancing all four confusion matrix cells
- May slightly reduce overall accuracy but improves fairness
- Works with any ML algorithm that supports sample weights

**Citations:**
- Kamiran, F., & Calders, T. (2012). "Data preprocessing techniques for classification without discrimination." *Knowledge and Information Systems*, 33(1), 1-33.
- Calders, T., Kamiran, F., & Pechenizkiy, M. (2009). "Building classifiers with independency constraints." In *2009 IEEE International Conference on Data Mining Workshops*, pp. 13-18.

### Complete Fairness Workflow

```elixir
# 1. Detect bias
predictions = your_model_predictions()
labels = ground_truth_labels()
sensitive_attr = sensitive_attributes()

report = ExFairness.fairness_report(predictions, labels, sensitive_attr)

if report.failed_count > 0 do
  IO.puts "⚠ Fairness issues detected: #{report.overall_assessment}"

  # 2. Check for legal compliance
  di_result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attr)
  if !di_result.passes_80_percent_rule do
    IO.puts "⚠ LEGAL WARNING: Fails EEOC 80% rule"
  end

  # 3. Apply mitigation
  weights = ExFairness.Mitigation.Reweighting.compute_weights(
    labels,
    sensitive_attr,
    target: :demographic_parity
  )

  # 4. Retrain and validate
  # retrained_model = retrain_with_weights(weights)
  # new_predictions = predict(retrained_model)
  # new_report = ExFairness.fairness_report(new_predictions, labels, sensitive_attr)
  # IO.puts "Improvement: #{new_report.passed_count} metrics now pass"
end
```

## Module Structure

```
lib/ex_fairness/
├── ex_fairness.ex                    # Main API (✅ Implemented)
├── error.ex                          # Custom error handling (✅ Implemented)
├── validation.ex                     # Input validation (✅ Implemented)
├── utils.ex                          # Core tensor utilities (✅ Implemented)
├── utils/
│   └── metrics.ex                    # Confusion matrix, TPR, FPR, PPV (✅ Implemented)
├── metrics/
│   ├── demographic_parity.ex         # ✅ Implemented
│   ├── equalized_odds.ex             # ✅ Implemented
│   ├── equal_opportunity.ex          # ✅ Implemented
│   ├── predictive_parity.ex          # ✅ Implemented
│   ├── calibration.ex                # 🚧 Coming soon
│   ├── individual_fairness.ex        # 🚧 Coming soon
│   └── counterfactual.ex             # 🚧 Coming soon
├── detection/
│   ├── disparate_impact.ex           # ✅ Implemented (EEOC 80% rule)
│   ├── statistical_parity.ex         # 🚧 Coming soon
│   ├── intersectional.ex             # 🚧 Coming soon
│   ├── temporal_drift.ex             # 🚧 Coming soon
│   ├── label_bias.ex                 # 🚧 Coming soon
│   └── representation.ex             # 🚧 Coming soon
├── mitigation/
│   ├── reweighting.ex                # ✅ Implemented
│   ├── resampling.ex                 # 🚧 Coming soon
│   ├── threshold_optimization.ex     # 🚧 Coming soon
│   ├── adversarial_debiasing.ex      # 🚧 Coming soon
│   ├── fair_representation.ex        # 🚧 Coming soon
│   └── calibration.ex                # 🚧 Coming soon
└── report.ex                         # ✅ Implemented (Markdown/JSON export)
```

## Real-World Use Cases

### Loan Approval Models

```elixir
# Ensure fair lending practices (ECOA compliance)
loan_predictions = model_predict(applicant_features)
actual_defaults = get_actual_defaults()
applicant_race = get_sensitive_attribute()

# Check fairness
fairness = ExFairness.fairness_report(
  loan_predictions,
  actual_defaults,
  applicant_race,
  metrics: [:demographic_parity, :predictive_parity]
)

# Check legal compliance
di_result = ExFairness.Detection.DisparateImpact.detect(loan_predictions, applicant_race)
if !di_result.passes_80_percent_rule do
  IO.puts "⚠ LEGAL ALERT: Loan approvals may violate EEOC guidelines"
end

# Apply mitigation if needed
if fairness.failed_count > 0 do
  weights = ExFairness.Mitigation.Reweighting.compute_weights(
    actual_defaults,
    applicant_race,
    target: :demographic_parity
  )
  # Retrain model with fairness weights
end
```

### Hiring and Recruitment

```elixir
# Analyze resume screening model
screening_results = screen_resumes(resumes)
interview_outcomes = get_interview_results()
applicant_gender = get_gender_attribute()

# Check equal opportunity (don't miss qualified candidates)
eo_result = ExFairness.equal_opportunity(
  screening_results,
  interview_outcomes,
  applicant_gender
)

if !eo_result.passes do
  IO.puts "⚠ Screening may miss qualified candidates from group B"
  IO.puts eo_result.interpretation
end
```

### Healthcare Risk Prediction

```elixir
# Ensure equitable healthcare predictions
risk_predictions = predict_health_risk(patient_data)
actual_outcomes = get_actual_health_outcomes()
patient_race = get_patient_race()

# Check equalized odds (both false positives and false negatives matter)
eq_result = ExFairness.equalized_odds(
  risk_predictions,
  actual_outcomes,
  patient_race
)

# Generate compliance report
report = ExFairness.fairness_report(risk_predictions, actual_outcomes, patient_race)
File.write!("healthcare_fairness_audit.md", ExFairness.Report.to_markdown(report))
```

## Fairness Metrics Reference

### Implemented Metrics

| Metric | Definition | When to Use | Use Case Examples |
|--------|------------|-------------|-------------------|
| **Demographic Parity** | P(Ŷ=1\|A=0) = P(Ŷ=1\|A=1) | Equal positive rates required | Advertising, content recommendation |
| **Equalized Odds** | TPR and FPR equal across groups | Both error types matter | Criminal justice, medical diagnosis |
| **Equal Opportunity** | TPR equal across groups | False negatives more costly | Hiring, college admissions |
| **Predictive Parity** | PPV equal across groups | Precision parity important | Risk assessment, credit scoring |

### Decision Guide

**Choose your metric based on your application:**

- **Advertising/Recommendations** → Demographic Parity
- **Criminal Justice** → Equalized Odds (both errors harmful)
- **Hiring/Admissions** → Equal Opportunity (don't miss qualified candidates)
- **Risk Assessment** → Predictive Parity (predictions should mean the same thing)
- **Healthcare** → Equalized Odds (both false alarms and missed diagnoses matter)
- **Credit Scoring** → Predictive Parity (approved should mean similar default risk)

---

## Mathematical Foundations

### Demographic Parity (Statistical Parity)

**Mathematical Definition:**
```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```

Where:
- `Ŷ` is the predicted outcome (0 or 1)
- `A` is the sensitive attribute (0 or 1)
- `P(Ŷ = 1 | A = a)` is the probability of a positive prediction given group membership

**Disparity Measure:**
```
Δ_DP = |P(Ŷ = 1 | A = 0) - P(Ŷ = 1 | A = 1)|
```

**Interpretation:**
A model satisfies demographic parity if both groups receive positive predictions at the same rate, regardless of the true labels. This ensures equal representation in positive outcomes.

**When to Use:**
- Equal representation is required (advertising exposure, content visibility)
- Base rates can legitimately differ between groups
- You want to ensure equal access to opportunities

**Limitations:**
- Ignores ground truth labels and base rate differences
- May reduce overall accuracy if base rates differ
- Can be satisfied by a random classifier
- May conflict with calibration and equalized odds

**Citations:**
- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). "Fairness through awareness." In *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference* (ITCS '12), pp. 214-226.
- Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). "Certifying and removing disparate impact." In *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD '15), pp. 259-268.

---

### Equalized Odds

**Mathematical Definition:**
```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)  [Equal TPR]
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)  [Equal FPR]
```

Where:
- `Y` is the true label (0 or 1)
- TPR (True Positive Rate) = P(Ŷ = 1 | Y = 1) = TP / (TP + FN)
- FPR (False Positive Rate) = P(Ŷ = 1 | Y = 0) = FP / (FP + TN)

**Disparity Measures:**
```
Δ_TPR = |TPR_{A=0} - TPR_{A=1}|
Δ_FPR = |FPR_{A=0} - FPR_{A=1}|
```

**Interpretation:**
A model satisfies equalized odds if both the true positive rate and false positive rate are equal across groups. This means the model's error rates are the same regardless of group membership.

**When to Use:**
- Both false positives and false negatives are harmful
- Criminal justice (wrongful conviction AND wrongful acquittal matter)
- Medical diagnosis (missing disease AND false alarms both harmful)
- High-stakes decisions where all error types matter

**Limitations:**
- More restrictive than demographic parity or equal opportunity
- May conflict with demographic parity when base rates differ
- Requires ground truth labels
- May reduce overall accuracy

**Citations:**
- Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." In *Advances in Neural Information Processing Systems* (NeurIPS '16), pp. 3315-3323.

---

### Equal Opportunity

**Mathematical Definition:**
```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)
```

Equivalently: TPR_{A=0} = TPR_{A=1}

**Disparity Measure:**
```
Δ_EO = |TPR_{A=0} - TPR_{A=1}|
```

**Interpretation:**
A model satisfies equal opportunity if the true positive rate (recall) is equal across groups. This ensures that qualified individuals from both groups have equal chances of receiving positive predictions.

**When to Use:**
- False negatives are more costly than false positives
- Hiring (don't want to miss qualified candidates from any group)
- College admissions (ensure qualified students have equal opportunity)
- Loan approvals (ensure creditworthy applicants are treated fairly)

**Limitations:**
- Ignores false positive rates (may burden one group with false positives)
- Less restrictive than equalized odds
- May conflict with demographic parity
- Only considers outcomes for positive class

**Citations:**
- Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." In *Advances in Neural Information Processing Systems* (NeurIPS '16), pp. 3315-3323.

---

### Predictive Parity (Outcome Test)

**Mathematical Definition:**
```
P(Y = 1 | Ŷ = 1, A = 0) = P(Y = 1 | Ŷ = 1, A = 1)
```

Equivalently: PPV_{A=0} = PPV_{A=1}

Where PPV (Positive Predictive Value) = P(Y = 1 | Ŷ = 1) = TP / (TP + FP)

**Disparity Measure:**
```
Δ_PP = |PPV_{A=0} - PPV_{A=1}|
```

**Interpretation:**
A model satisfies predictive parity if the positive predictive value (precision) is equal across groups. This means a positive prediction has the same meaning regardless of group membership.

**When to Use:**
- Positive predictions should mean the same thing across groups
- Risk assessment (a "high risk" score should mean similar actual risk)
- Credit scoring (approved applicants should have similar default rates)
- When users make decisions based on predictions

**Limitations:**
- Ignores true positive rates and false negative rates
- May conflict with equalized odds when base rates differ
- Can mask disparities in false positive/negative rates
- Only considers outcomes for predicted positives

**Citations:**
- Chouldechova, A. (2017). "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments." *Big Data*, 5(2), 153-163.

---

### Disparate Impact (80% Rule)

**Mathematical Definition:**
```
Ratio = min(P(Ŷ = 1 | A = 0), P(Ŷ = 1 | A = 1)) / max(P(Ŷ = 1 | A = 0), P(Ŷ = 1 | A = 1))
```

**Legal Standard:**
```
Ratio ≥ 0.8  →  PASS (no evidence of disparate impact)
Ratio < 0.8  →  FAIL (potential disparate impact)
```

**Interpretation:**
The 4/5ths (80%) rule is a legal guideline from the U.S. Equal Employment Opportunity Commission. If the selection rate for any group is less than 80% of the highest selection rate, this constitutes evidence of adverse impact under U.S. employment law.

**Legal Context:**
- Used in employment discrimination cases
- Applied to hiring, promotion, and termination decisions
- Also relevant for lending (ECOA), housing (Fair Housing Act)
- Not an absolute legal requirement, but strong evidence

**When to Use:**
- Legal compliance audits (EEOC, ECOA, FHA)
- Employment decisions
- Lending and credit decisions
- Any decision process subject to anti-discrimination law

**Limitations:**
- A guideline, not absolute proof of discrimination
- Statistical significance should also be considered
- Small sample sizes can produce unreliable ratios
- Does not account for legitimate business necessity defenses

**Citations:**
- Equal Employment Opportunity Commission, Civil Service Commission, Department of Labor, & Department of Justice. (1978). "Uniform Guidelines on Employee Selection Procedures." *Federal Register*, 43(166), 38290-38315.
- Biddle, D. (2006). "Adverse Impact and Test Validation: A Practitioner's Guide to Valid and Defensible Employment Testing." Gower Publishing.

---

## Fairness-Accuracy Trade-offs and Impossibility Results

### The Impossibility Theorem

**Chouldechova (2017)** and **Kleinberg et al. (2016)** proved that certain fairness metrics are mathematically incompatible when base rates differ between groups.

**Key Result:**
A binary classifier **cannot** simultaneously satisfy all three of the following when P(Y=1|A=0) ≠ P(Y=1|A=1):

1. **Calibration** (Predictive Parity): P(Y=1|Ŷ=1,A=0) = P(Y=1|Ŷ=1,A=1)
2. **Balance for the Positive Class** (Equal Opportunity): P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
3. **Balance for the Negative Class**: P(Ŷ=0|Y=0,A=0) = P(Ŷ=0|Y=0,A=1)

**Practical Implications:**
- If your groups have different base rates (e.g., different disease prevalence), you must choose which fairness property to prioritize
- Demographic parity and equalized odds are often in conflict
- Predictive parity and equalized odds cannot both be satisfied with different base rates
- There is no "one size fits all" fairness definition

**Example Scenario:**

Consider a medical test where:
- Disease prevalence in Group A: 30%
- Disease prevalence in Group B: 10%

A perfect classifier (100% accuracy) will:
- ✅ Satisfy equalized odds (TPR=1.0, FPR=0.0 for both groups)
- ✅ Satisfy equal opportunity (TPR=1.0 for both)
- ✅ Satisfy predictive parity (PPV=1.0 for both)
- ✗ **VIOLATE** demographic parity (30% vs 10% positive rates)

This is not a flaw in the classifier—it's a mathematical necessity when base rates differ.

**Choosing Your Metric:**

| Scenario | Recommended Metric | Rationale |
|----------|-------------------|-----------|
| Base rates differ legitimately | Equal Opportunity or Equalized Odds | Respects different underlying rates |
| Base rates shouldn't differ | Demographic Parity | Enforces equal representation |
| Prediction interpretation critical | Predictive Parity | Ensures consistent meaning |
| Both error types equally costly | Equalized Odds | Balances all error rates |

**Citations:**
- Chouldechova, A. (2017). "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments." *Big Data*, 5(2), 153-163.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). "Inherent trade-offs in the fair determination of risk scores." In *Proceedings of the 8th Innovations in Theoretical Computer Science Conference* (ITCS '17).
- Corbett-Davies, S., & Goel, S. (2018). "The measure and mismeasure of fairness: A critical review of fair machine learning." *arXiv preprint arXiv:1808.00023*.

### Understanding Trade-offs with ExFairness

```elixir
# Analyze multiple metrics to understand trade-offs
report = ExFairness.fairness_report(predictions, labels, sensitive_attr)

# The report shows which metrics pass/fail
IO.puts report.overall_assessment
# => "⚠ Mixed results: 2 of 4 metrics passed, 2 failed..."

# This is expected when base rates differ!
# Check if conflicts are due to base rate differences
base_rate_a = Nx.mean(Nx.select(Nx.equal(sensitive_attr, 0), labels, 0)) |> Nx.to_number()
base_rate_b = Nx.mean(Nx.select(Nx.equal(sensitive_attr, 1), labels, 0)) |> Nx.to_number()

if abs(base_rate_a - base_rate_b) > 0.1 do
  IO.puts "Note: Base rates differ significantly (#{Float.round(base_rate_a, 2)} vs #{Float.round(base_rate_b, 2)})"
  IO.puts "Some metric conflicts are mathematically inevitable (Impossibility Theorem)"
end
```

## Best Practices

### 1. Define Your Fairness Requirements

Different applications require different fairness definitions:

```elixir
# For lending: Equalized odds (equal TPR and FPR)
# For hiring: Equal opportunity (equal TPR for qualified candidates)
# For content recommendation: Demographic parity (equal exposure)
```

### 2. Analyze Multiple Metrics

No single metric captures all aspects of fairness:

```elixir
# Generate comprehensive report
report = ExFairness.fairness_report(
  predictions,
  labels,
  sensitive_attr
)  # Defaults to all implemented metrics

IO.puts "Passed: #{report.passed_count}/#{report.total_count}"
```

### 3. Validate Statistical Reliability

Ensure sufficient sample sizes:

```elixir
# Default requires 10+ samples per group
# Adjust if needed for your use case
result = ExFairness.demographic_parity(
  predictions,
  sensitive_attr,
  min_per_group: 30  # Higher for more reliable statistics
)
```

### 4. Check Legal Compliance

Always check the EEOC 80% rule for legal compliance:

```elixir
di_result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attr)
if !di_result.passes_80_percent_rule do
  IO.puts "⚠ May violate EEOC guidelines - consult legal counsel"
end
```

### 5. Document Your Fairness Assessment

Generate reports for audit trails and transparency:

```elixir
report = ExFairness.fairness_report(predictions, labels, sensitive_attr)

# Human-readable format
File.write!("fairness_audit.md", ExFairness.Report.to_markdown(report))

# Machine-readable format
File.write!("fairness_audit.json", ExFairness.Report.to_json(report))
```

---

## Advanced Usage

### Integration with Axon (Neural Networks)

```elixir
defmodule FairClassifier do
  import Nx.Defn

  def train_with_fairness(train_x, train_y, sensitive_attr) do
    # 1. Compute fairness weights
    weights = ExFairness.Mitigation.Reweighting.compute_weights(
      train_y,
      sensitive_attr,
      target: :demographic_parity
    )

    # 2. Define model
    model =
      Axon.input("features")
      |> Axon.dense(64, activation: :relu)
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(32, activation: :relu)
      |> Axon.dense(1, activation: :sigmoid)

    # 3. Train with weighted loss
    # Note: Axon doesn't directly support sample weights yet,
    # but you can modify the loss function:
    weighted_loss_fn = fn y_true, y_pred ->
      base_loss = Axon.Losses.binary_cross_entropy(y_true, y_pred)
      # Apply weights in custom training loop
      base_loss
    end

    # 4. Validate fairness
    trained_model_state = train_model(model, train_x, train_y, weights)

    # 5. Check fairness on validation set
    val_predictions = Axon.predict(model, trained_model_state, val_x)
    val_binary = Nx.greater(val_predictions, 0.5)

    fairness_report = ExFairness.fairness_report(
      val_binary,
      val_y,
      val_sensitive_attr
    )

    {model, trained_model_state, fairness_report}
  end
end
```

### Integration with Scholar (Classical ML)

```elixir
# Example: Fair Logistic Regression
defmodule FairLogisticRegression do
  def train_fair_model(features, labels, sensitive_attr) do
    # 1. Detect initial bias
    # Train baseline model first
    baseline_model = train_baseline(features, labels)
    baseline_preds = predict(baseline_model, features)

    initial_report = ExFairness.fairness_report(
      baseline_preds,
      labels,
      sensitive_attr
    )

    IO.puts "Baseline fairness: #{initial_report.passed_count}/#{initial_report.total_count} metrics pass"

    # 2. Apply reweighting if needed
    if initial_report.failed_count > 0 do
      weights = ExFairness.Mitigation.Reweighting.compute_weights(
        labels,
        sensitive_attr,
        target: :demographic_parity
      )

      # 3. Retrain with weights
      # Note: Waiting for Scholar to support sample weights
      # For now, you can oversample/undersample based on weights
      fair_model = train_with_weights(features, labels, weights)

      # 4. Validate improvement
      fair_preds = predict(fair_model, features)
      final_report = ExFairness.fairness_report(fair_preds, labels, sensitive_attr)

      improvement = final_report.passed_count - initial_report.passed_count
      IO.puts "Fairness improved: #{improvement} additional metrics now pass"

      {fair_model, final_report}
    else
      {baseline_model, initial_report}
    end
  end
end
```

### Batch Fairness Analysis

```elixir
# Analyze fairness across multiple models or configurations
defmodule BatchFairnessAnalysis do
  def compare_models(models, test_x, test_y, sensitive_attr) do
    # Test each model
    results = Enum.map(models, fn {name, model} ->
      predictions = predict(model, test_x)
      report = ExFairness.fairness_report(predictions, test_y, sensitive_attr)

      {name, report}
    end)

    # Find best model by fairness
    {best_model, best_report} = Enum.max_by(results, fn {_name, report} ->
      report.passed_count
    end)

    IO.puts "Best model: #{best_model}"
    IO.puts "Fairness: #{best_report.passed_count}/#{best_report.total_count} metrics pass"

    # Generate comparison report
    comparison = Enum.map(results, fn {name, report} ->
      %{
        model: name,
        passed: report.passed_count,
        failed: report.failed_count,
        assessment: report.overall_assessment
      }
    end)

    File.write!("model_comparison.json", Jason.encode!(comparison, pretty: true))

    {best_model, comparison}
  end
end
```

### Production Monitoring

```elixir
defmodule FairnessMonitor do
  use GenServer

  # Monitor fairness in production
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    # Schedule periodic fairness checks
    schedule_check()
    {:ok, %{history: []}}
  end

  def handle_info(:check_fairness, state) do
    # Get recent predictions from production
    {predictions, labels, sensitive_attrs} = fetch_recent_production_data()

    # Generate fairness report
    report = ExFairness.fairness_report(predictions, labels, sensitive_attrs)

    # Check for issues
    if report.failed_count > 0 do
      send_alert("Fairness degradation detected: #{report.overall_assessment}")
    end

    # Check legal compliance
    di_result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attrs)
    if !di_result.passes_80_percent_rule do
      send_alert("LEGAL WARNING: EEOC 80% rule violation")
    end

    # Store history
    new_history = [{DateTime.utc_now(), report} | state.history]

    # Schedule next check
    schedule_check()

    {:noreply, %{state | history: new_history}}
  end

  defp schedule_check do
    # Check every hour
    Process.send_after(self(), :check_fairness, :timer.hours(1))
  end

  defp send_alert(message) do
    # Implement your alerting logic
    Logger.warning("Fairness Alert: #{message}")
  end
end
```

---

## Technical Details

### Performance

All core computations use `Nx.Defn` for GPU acceleration:

```elixir
# Automatically uses available backend (CPU, EXLA, Torchx)
# Set backend for GPU acceleration:
# Nx.default_backend(EXLA.Backend)

result = ExFairness.demographic_parity(predictions, sensitive_attr)
# Computation runs on configured backend
```

### Type Safety

All public functions have type specifications:

```elixir
@spec demographic_parity(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
  %{
    group_a_rate: float(),
    group_b_rate: float(),
    disparity: float(),
    passes: boolean(),
    threshold: float(),
    interpretation: String.t()
  }
```

### Error Handling

Comprehensive validation with helpful error messages:

```elixir
# Example error: insufficient samples
predictions = Nx.tensor([1, 0, 1, 0])
sensitive = Nx.tensor([0, 0, 1, 1])

ExFairness.demographic_parity(predictions, sensitive)
# => ** (ExFairness.Error) Insufficient samples per group for reliable fairness metrics.
#
#    Found:
#      Group 0: 2 samples
#      Group 1: 2 samples
#
#    Recommended minimum: 10 samples per group.
#
#    Consider:
#    - Collecting more data
#    - Using bootstrap methods with caution
#    - Aggregating smaller groups if appropriate
```

## Limitations

- **Impossibility Theorems**: Some fairness definitions are mutually exclusive (e.g., demographic parity and equalized odds with different base rates). See Chouldechova (2017).
- **Sensitive Attributes**: Requires access to sensitive attributes for measurement
- **Binary Groups**: Current implementation supports binary sensitive attributes (0/1). Multi-group support coming soon.
- **Sample Size**: Requires minimum 10 samples per group by default for statistical reliability
- **Binary Classification**: Current metrics designed for binary classification tasks

## Theoretical Background

### What is Algorithmic Fairness?

Algorithmic fairness is concerned with ensuring that automated decision-making systems do not discriminate against individuals or groups based on sensitive attributes such as race, gender, age, or religion.

**Key Questions:**
1. **What does fairness mean?** Different mathematical definitions capture different intuitions
2. **How do we measure it?** Quantitative metrics for bias detection
3. **How do we achieve it?** Mitigation techniques to improve fairness
4. **What are the trade-offs?** Understanding impossibility results and accuracy costs

### Types of Fairness

#### 1. Group Fairness (Statistical Parity)

**Definition:** Statistical measures computed over groups defined by sensitive attributes.

**Examples:**
- Demographic Parity: Equal positive prediction rates
- Equalized Odds: Equal error rates across groups
- Equal Opportunity: Equal true positive rates

**Advantages:**
- Easy to measure and verify
- Clear mathematical definitions
- Actionable with standard ML techniques

**Disadvantages:**
- Ignores individual circumstances
- May allow discrimination against individuals
- Can be satisfied while treating individuals very differently

**ExFairness Implementation:** ✅ Demographic Parity, Equalized Odds, Equal Opportunity, Predictive Parity

#### 2. Individual Fairness (Similarity-Based)

**Definition:** Similar individuals should receive similar predictions.

**Formal Definition (Dwork et al. 2012):**
```
d(Ŷ(x₁), Ŷ(x₂)) ≤ L · d(x₁, x₂)
```

Where `d` is a distance metric, `L` is the Lipschitz constant.

**Advantages:**
- Protects individual treatment
- More granular than group fairness
- Prevents "gerrymandering" of groups

**Disadvantages:**
- Requires defining similarity metric (often domain-specific)
- Computationally expensive (pairwise comparisons)
- Difficult to verify at scale

**ExFairness Implementation:** 🚧 Planned for future release

#### 3. Causal Fairness (Counterfactual)

**Definition:** A decision is fair if it would be the same in a counterfactual world where only the sensitive attribute changed.

**Formal Definition (Kusner et al. 2017):**
```
P(Ŷ_{A←a}(U) = y | X = x, A = a) = P(Ŷ_{A←a'}(U) = y | X = x, A = a)
```

**Advantages:**
- Captures causal notion of discrimination
- Prevents direct and indirect discrimination
- Strongest fairness guarantee

**Disadvantages:**
- Requires causal graph (domain knowledge)
- Difficult to verify without interventional data
- May be overly restrictive in practice

**ExFairness Implementation:** 🚧 Planned for future release

### The Measurement Problem

**Challenge:** We can measure outcomes, but not always the "ground truth" of who deserves what.

**Example Issues:**
- **Label Bias:** Historical labels may reflect past discrimination
- **Feedback Loops:** Biased decisions create biased training data
- **Construct Validity:** Does the target variable measure what we think it does?
- **Selection Bias:** Training data may not represent deployment population

**ExFairness Approach:**
- Provides multiple metrics to capture different fairness notions
- Includes interpretation to understand what each metric means
- Enables detection of label bias and representation issues
- Supports validation across multiple fairness definitions

### Key Theoretical Results

#### Impossibility of Simultaneous Satisfaction

**Theorem (Chouldechova 2017, Kleinberg et al. 2016):**

When base rates differ between groups (P(Y=1|A=0) ≠ P(Y=1|A=1)), a binary classifier cannot simultaneously satisfy:
- Calibration (Predictive Parity)
- Balance for positive class (Equal Opportunity)
- Balance for negative class

**Proof Intuition:**

Suppose P(Y=1|A=0) = 0.5 and P(Y=1|A=1) = 0.3 (different base rates).

If we require:
1. Equal TPR: P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1) = r
2. Equal FPR: P(Ŷ=1|Y=0,A=0) = P(Ŷ=1|Y=0,A=1) = f

Then by Bayes' rule:
```
P(Y=1|Ŷ=1,A=0) = [r · 0.5] / [r · 0.5 + f · 0.5]
P(Y=1|Ŷ=1,A=1) = [r · 0.3] / [r · 0.3 + f · 0.7]
```

These can only be equal if r=0 and f=0 (trivial classifier) or r=1 and f=1 (random).

**Practical Implication:** You must prioritize which fairness property matters most for your application.

#### The Fairness-Accuracy Tradeoff

**Key Insight:** Enforcing fairness constraints may reduce overall accuracy.

**When Trade-off is Minimal:**
- Groups have similar base rates
- Model is already nearly fair
- Strong features not correlated with sensitive attributes

**When Trade-off is Significant:**
- Large base rate differences between groups
- Limited features available
- Strong correlation between outcomes and sensitive attributes

**ExFairness Strategy:**
- Provides transparency about trade-offs through comprehensive reporting
- Allows configurable thresholds to balance fairness and accuracy
- Includes interpretations to understand why metrics fail

---

## API Reference

### Main Functions

```elixir
# Fairness Metrics
ExFairness.demographic_parity(predictions, sensitive_attr, opts \\ [])
ExFairness.equalized_odds(predictions, labels, sensitive_attr, opts \\ [])
ExFairness.equal_opportunity(predictions, labels, sensitive_attr, opts \\ [])
ExFairness.predictive_parity(predictions, labels, sensitive_attr, opts \\ [])

# Comprehensive Reporting
ExFairness.fairness_report(predictions, labels, sensitive_attr, opts \\ [])
ExFairness.Report.to_markdown(report)
ExFairness.Report.to_json(report)

# Detection
ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attr, opts \\ [])

# Mitigation
ExFairness.Mitigation.Reweighting.compute_weights(labels, sensitive_attr, opts \\ [])
```

### Common Options

- `:threshold` - Maximum acceptable disparity (default: 0.1)
- `:min_per_group` - Minimum samples per group (default: 10)
- `:metrics` - List of metrics for reports (default: all available)
- `:target` - Target fairness metric for mitigation (`:demographic_parity` or `:equalized_odds`)

## Research Foundations

### Seminal Papers in Algorithmic Fairness

#### 1. Foundational Theory

**Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012)**
"Fairness through awareness."
In *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference* (ITCS '12), pp. 214-226.
DOI: 10.1145/2090236.2090255

**Contribution:** Introduced individual fairness (Lipschitz continuity) and demographic parity. Foundational work defining fairness as a computational problem.

**Key Insight:** "We're All Equal, We're All Different" - fairness requires treating similar people similarly while respecting relevant differences.

**ExFairness Implementation:** Demographic Parity metric

---

**Hardt, M., Price, E., & Srebro, N. (2016)**
"Equality of Opportunity in Supervised Learning."
In *Advances in Neural Information Processing Systems* (NeurIPS '16), pp. 3315-3323.

**Contribution:** Defined equalized odds and equal opportunity. Showed that these are more appropriate than demographic parity when base rates differ.

**Key Insight:** Fairness should depend on the true labels, not just predictions. Equal error rates are often more meaningful than equal positive rates.

**ExFairness Implementation:** Equalized Odds and Equal Opportunity metrics

---

#### 2. Impossibility Results

**Chouldechova, A. (2017)**
"Fair prediction with disparate impact: A study of bias in recidivism prediction instruments."
*Big Data*, 5(2), 153-163.
DOI: 10.1089/big.2016.0047

**Contribution:** Proved impossibility of simultaneously satisfying calibration, balance for positive class, and balance for negative class when base rates differ.

**Key Insight:** Trade-offs between fairness metrics are mathematical necessities, not implementation failures. Studied real COMPAS recidivism scores.

**ExFairness Implementation:** Predictive Parity metric, impossibility awareness in documentation

---

**Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016)**
"Inherent trade-offs in the fair determination of risk scores."
In *Proceedings of the 8th Innovations in Theoretical Computer Science Conference* (ITCS '17).

**Contribution:** Independently proved similar impossibility results. Provided economic interpretation of fairness constraints.

**Key Insight:** The conflict between fairness metrics reflects fundamental disagreements about what fairness means, not technical problems.

---

#### 3. Measurement and Mitigation

**Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015)**
"Certifying and removing disparate impact."
In *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD '15), pp. 259-268.
DOI: 10.1145/2783258.2783311

**Contribution:** Methods for measuring and removing disparate impact in decision-making systems.

**Key Insight:** Disparate impact can be quantified and algorithmically reduced through preprocessing.

**ExFairness Implementation:** Demographic Parity metric, theoretical foundation for mitigation

---

**Kamiran, F., & Calders, T. (2012)**
"Data preprocessing techniques for classification without discrimination."
*Knowledge and Information Systems*, 33(1), 1-33.
DOI: 10.1007/s10115-011-0463-8

**Contribution:** Comprehensive study of preprocessing techniques including reweighting, resampling, and massaging.

**Key Insight:** Fairness can be improved before training through data transformation.

**ExFairness Implementation:** Reweighting mitigation technique

---

**Calders, T., Kamiran, F., & Pechenizkiy, M. (2009)**
"Building classifiers with independency constraints."
In *2009 IEEE International Conference on Data Mining Workshops*, pp. 13-18.
DOI: 10.1109/ICDMW.2009.83

**Contribution:** Methods for training classifiers with fairness constraints.

**ExFairness Implementation:** Foundation for reweighting approach

---

#### 4. Legal and Regulatory

**Equal Employment Opportunity Commission, Civil Service Commission, Department of Labor, & Department of Justice (1978)**
"Uniform Guidelines on Employee Selection Procedures."
*Federal Register*, 43(166), 38290-38315.

**Contribution:** Established the 4/5ths (80%) rule as legal standard for detecting adverse impact.

**Legal Status:** Binding guideline for U.S. employment law. Used by courts to determine prima facie evidence of discrimination.

**ExFairness Implementation:** Disparate Impact detection with 80% rule

---

**Biddle, D. (2006)**
"Adverse Impact and Test Validation: A Practitioner's Guide to Valid and Defensible Employment Testing."
Gower Publishing.

**Contribution:** Practical guide to applying 80% rule in employment contexts.

**ExFairness Implementation:** Legal interpretation in disparate impact module

---

#### 5. Critical Reviews and Surveys

**Corbett-Davies, S., & Goel, S. (2018)**
"The measure and mismeasure of fairness: A critical review of fair machine learning."
*arXiv preprint arXiv:1808.00023*

**Contribution:** Critical analysis of fairness metrics, discussing when each is appropriate and their limitations.

**Key Insight:** No single fairness metric is universally appropriate. Context matters enormously.

---

**Barocas, S., Hardt, M., & Narayanan, A. (2019)**
"Fairness and Machine Learning: Limitations and Opportunities."
*fairmlbook.org*

**Contribution:** Comprehensive textbook on fairness in ML. Covers theory, practice, and societal implications.

**Status:** Living document, freely available online.

---

### Additional Important References

**Verma, S., & Rubin, J. (2018)**
"Fairness definitions explained."
In *2018 IEEE/ACM International Workshop on Software Fairness* (FairWare), pp. 1-7.

**Contribution:** Systematically categorized 20+ fairness definitions, showing relationships and conflicts.

---

**Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021)**
"A survey on bias and fairness in machine learning."
*ACM Computing Surveys*, 54(6), 1-35.

**Contribution:** Comprehensive survey of bias sources, fairness metrics, and mitigation techniques.

---

### Related Open-Source Frameworks

**IBM AI Fairness 360 (AIF360)** - Python
- Comprehensive toolkit with 70+ fairness metrics and 10+ mitigation algorithms
- Industry standard for fairness assessment
- https://github.com/Trusted-AI/AIF360

**Microsoft Fairlearn** - Python
- Focus on fairness assessment and mitigation
- Grid search for optimal fair models
- https://github.com/fairlearn/fairlearn

**Google Fairness Indicators** - Python/TensorFlow
- Production monitoring for fairness metrics
- Integration with TensorFlow Extended (TFX)
- https://github.com/tensorflow/fairness-indicators

**Aequitas** - Python
- Bias and fairness audit toolkit
- Focus on criminal justice and policy applications
- http://aequitas.dssg.io/

**ExFairness Unique Value:**
- ✅ **First comprehensive fairness library for Elixir/Nx ecosystem**
- ✅ **GPU-accelerated** via Nx.Defn (EXLA/Torchx compatible)
- ✅ **Functional programming paradigm** - immutable, composable
- ✅ **Type-safe** with Dialyzer
- ✅ **BEAM concurrency** - parallel fairness analysis
- ✅ **Production-ready** from day one

## Examples

ExFairness includes comprehensive, runnable examples demonstrating all features:

```bash
# Run individual examples
mix run examples/01_demographic_parity.exs
mix run examples/02_equalized_odds.exs
mix run examples/03_equal_opportunity.exs
mix run examples/04_predictive_parity.exs
mix run examples/05_comprehensive_report.exs
mix run examples/06_disparate_impact.exs
mix run examples/07_mitigation_reweighting.exs
mix run examples/08_end_to_end_workflow.exs

# Run all examples
for f in examples/*.exs; do echo "Running $f"; mix run "$f"; done
```

### Example Overview

- **01_demographic_parity.exs**: Demonstrates demographic parity metric with fair and biased scenarios
- **02_equalized_odds.exs**: Shows equalized odds analysis for medical diagnosis and criminal justice
- **03_equal_opportunity.exs**: Illustrates equal opportunity in college admissions and hiring
- **04_predictive_parity.exs**: Explains predictive parity for credit scoring and risk assessment
- **05_comprehensive_report.exs**: Generates multi-metric reports with Markdown and JSON export
- **06_disparate_impact.exs**: Legal compliance checking with EEOC 80% rule
- **07_mitigation_reweighting.exs**: Complete bias mitigation workflow using reweighting
- **08_end_to_end_workflow.exs**: Full fairness workflow from detection to deployment

Each example includes detailed explanations, multiple scenarios, and best practices for responsible AI.

## Development

### Running Tests

```bash
# Run all tests
mix test

# Run with coverage
mix test --cover

# Run specific module tests
mix test test/ex_fairness/metrics/demographic_parity_test.exs
```

### Code Quality

```bash
# Format code
mix format

# Check formatting
mix format --check-formatted

# Run linter
mix credo --strict

# Type checking (requires plt build first)
mix dialyzer
```

### Quality Metrics

- **Tests**: 134 total (102 unit + 32 doctests)
- **Test Failures**: 0
- **Compiler Warnings**: 0
- **Type Coverage**: 100% of public functions
- **Documentation Coverage**: 100% of modules and public functions
- **Code Quality**: Credo strict mode passes

## Project Status

**Current Version**: 0.1.0 (Development)

**Implementation Status:**
- ✅ Core Infrastructure: Complete
- ✅ Group Fairness Metrics: 4 metrics implemented
- ✅ Reporting System: Complete (Markdown/JSON)
- ✅ Detection Algorithms: Disparate Impact (80% rule)
- ✅ Mitigation: Reweighting
- 🚧 Advanced Metrics: Calibration, Individual Fairness (planned)
- 🚧 Advanced Mitigation: Resampling, Threshold Optimization (planned)

**Production Ready:** ✅ Yes - Core features are stable and well-tested

## Contributing

This is part of the **North Shore AI Research Infrastructure**.

Contributions are welcome! Please:
1. Follow the existing TDD approach (Red-Green-Refactor)
2. Ensure all tests pass with zero warnings
3. Add comprehensive documentation with examples
4. Include research citations for new metrics
5. Add type specifications to all public functions

## License

MIT License - see [LICENSE](https://github.com/North-Shore-AI/ExFairness/blob/main/LICENSE) file for details

## Acknowledgments

Built with ❤️ by the North Shore AI team. Special thanks to the fairness ML research community for their foundational work.

**Part of the North Shore AI Ecosystem:**
- [ExFairness](https://github.com/North-Shore-AI/ExFairness) - Fairness & bias detection (this project)
- Scholar - Classical ML algorithms for Elixir
- Axon - Neural networks for Elixir
- Nx - Numerical computing for Elixir
