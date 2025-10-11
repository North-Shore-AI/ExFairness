<p align="center">
  <img src="assets/ExFairness.svg" alt="ExFairness" width="150"/>
</p>

# ExFairness

**Fairness and Bias Detection Library for Elixir AI/ML Systems**

[![Elixir](https://img.shields.io/badge/elixir-1.14+-purple.svg)](https://elixir-lang.org)
[![OTP](https://img.shields.io/badge/otp-25+-red.svg)](https://www.erlang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/North-Shore-AI/ExFairness/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-blueviolet.svg)](https://hexdocs.pm/ex_fairness)

---

ExFairness is a comprehensive library for detecting, measuring, and mitigating bias in AI/ML systems built with Elixir. It provides rigorous fairness metrics, bias detection algorithms, and mitigation techniques to ensure your models make equitable predictions across different demographic groups.

## Features

### Fairness Metrics

- **Demographic Parity**: Ensures equal positive prediction rates across groups
- **Equalized Odds**: Ensures equal true positive and false positive rates across groups
- **Equal Opportunity**: Ensures equal true positive rates across groups (recall parity)
- **Predictive Parity**: Ensures equal positive predictive values (precision parity)
- **Calibration**: Ensures predicted probabilities match actual outcomes across groups
- **Individual Fairness**: Ensures similar individuals receive similar predictions
- **Counterfactual Fairness**: Predictions remain unchanged under counterfactual changes to sensitive attributes

### Bias Detection

- **Statistical Parity Testing**: Detects disparate impact in model outcomes
- **Disparate Impact Analysis**: Measures the ratio of outcomes between groups
- **Simpson's Paradox Detection**: Identifies misleading aggregated statistics
- **Subgroup Analysis**: Analyzes fairness across intersectional groups
- **Temporal Bias Drift**: Monitors fairness metrics over time
- **Label Bias Detection**: Identifies bias in training labels
- **Representation Bias**: Measures data imbalance across groups

### Mitigation Techniques

- **Pre-processing**: Reweighting, resampling, and fair representation learning
- **In-processing**: Adversarial debiasing, fairness constraints during training
- **Post-processing**: Threshold optimization, calibration, and reject option classification
- **Fair Feature Selection**: Identifies and removes biased features
- **Fairness-Aware Ensemble Methods**: Combines models to improve fairness

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
    {:ex_fairness, "~> 0.1.0"}
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
predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 0 = group A, 1 = group B

result = ExFairness.demographic_parity(predictions, sensitive_attr)
# => %{
#   group_a_rate: 0.50,
#   group_b_rate: 0.75,
#   disparity: 0.25,
#   passes: false,
#   threshold: 0.10
# }
```

### Measure Equalized Odds

```elixir
# Include ground truth labels
predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1])
labels = Nx.tensor([1, 0, 1, 0, 0, 1, 1, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])

result = ExFairness.equalized_odds(predictions, labels, sensitive_attr)
# => %{
#   group_a_tpr: 0.67,
#   group_b_tpr: 0.67,
#   group_a_fpr: 0.50,
#   group_b_fpr: 0.00,
#   tpr_disparity: 0.00,
#   fpr_disparity: 0.50,
#   passes: false
# }
```

### Equal Opportunity Analysis

```elixir
result = ExFairness.equal_opportunity(predictions, labels, sensitive_attr)
# => %{
#   group_a_tpr: 0.67,
#   group_b_tpr: 0.67,
#   disparity: 0.00,
#   passes: true,
#   interpretation: "Model provides equal opportunity across groups"
# }
```

### Comprehensive Fairness Report

```elixir
report = ExFairness.fairness_report(predictions, labels, sensitive_attr,
  metrics: [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity]
)

# => %{
#   demographic_parity: %{passes: false, disparity: 0.25},
#   equalized_odds: %{passes: false, tpr_disparity: 0.00, fpr_disparity: 0.50},
#   equal_opportunity: %{passes: true, disparity: 0.00},
#   predictive_parity: %{passes: true, disparity: 0.05},
#   overall_assessment: "2 of 4 fairness metrics passed",
#   recommendations: ["Consider threshold optimization for demographic parity", ...]
# }
```

## Bias Detection

### Disparate Impact Analysis

```elixir
# Measure the 80% rule (4/5ths rule)
impact = ExFairness.disparate_impact(predictions, sensitive_attr)
# => %{
#   ratio: 0.67,
#   passes_80_percent_rule: false,
#   interpretation: "Group B receives 67% the rate of positive outcomes as Group A"
# }
```

### Intersectional Fairness

```elixir
# Analyze fairness across multiple sensitive attributes
gender = Nx.tensor([0, 0, 1, 1, 0, 0, 1, 1])
race = Nx.tensor([0, 1, 0, 1, 0, 1, 0, 1])

result = ExFairness.intersectional_fairness(predictions, labels,
  sensitive_attrs: [gender, race],
  attr_names: ["gender", "race"]
)

# Analyzes all combinations: (male, white), (male, black), (female, white), (female, black)
```

### Temporal Bias Monitoring

```elixir
# Monitor fairness drift over time
metrics_history = [
  {~D[2025-01-01], %{demographic_parity: 0.10}},
  {~D[2025-02-01], %{demographic_parity: 0.15}},
  {~D[2025-03-01], %{demographic_parity: 0.25}}
]

drift = ExFairness.temporal_drift(metrics_history, threshold: 0.05)
# => %{
#   drift_detected: true,
#   drift_magnitude: 0.15,
#   alert_level: :warning
# }
```

## Mitigation Techniques

### Reweighting (Pre-processing)

```elixir
# Reweight training samples to achieve demographic parity
{reweighted_data, weights} = ExFairness.Mitigation.reweight(
  training_data,
  sensitive_attr,
  target: :demographic_parity
)
```

### Threshold Optimization (Post-processing)

```elixir
# Find optimal decision thresholds per group
probabilities = Nx.tensor([0.3, 0.7, 0.8, 0.6, 0.4, 0.9, 0.5, 0.7])

thresholds = ExFairness.Mitigation.optimize_thresholds(
  probabilities,
  labels,
  sensitive_attr,
  target_metric: :equalized_odds
)
# => %{group_a_threshold: 0.55, group_b_threshold: 0.45}

# Apply group-specific thresholds
fair_predictions = ExFairness.Mitigation.apply_thresholds(
  probabilities,
  sensitive_attr,
  thresholds
)
```

### Adversarial Debiasing (In-processing)

```elixir
# Train with fairness constraints (integrates with Axon)
model = ExFairness.Mitigation.adversarial_debiasing(
  base_model,
  sensitive_attr,
  adversary_strength: 0.5
)
```

### Fair Representation Learning

```elixir
# Learn fair representations that remove sensitive information
{fair_features, encoder} = ExFairness.Mitigation.fair_representation(
  features,
  sensitive_attr,
  method: :variational_fair_autoencoder
)
```

## Module Structure

```
lib/ex_fairness/
├── ex_fairness.ex                    # Main API
├── metrics/
│   ├── demographic_parity.ex         # Demographic parity metrics
│   ├── equalized_odds.ex             # Equalized odds metrics
│   ├── equal_opportunity.ex          # Equal opportunity metrics
│   ├── predictive_parity.ex          # Predictive parity metrics
│   ├── calibration.ex                # Calibration metrics
│   ├── individual_fairness.ex        # Individual fairness metrics
│   └── counterfactual.ex             # Counterfactual fairness
├── detection/
│   ├── disparate_impact.ex           # Disparate impact analysis
│   ├── statistical_parity.ex         # Statistical parity testing
│   ├── intersectional.ex             # Intersectional analysis
│   ├── temporal_drift.ex             # Temporal bias monitoring
│   ├── label_bias.ex                 # Label bias detection
│   └── representation.ex             # Representation bias
├── mitigation/
│   ├── reweighting.ex                # Reweighting techniques
│   ├── resampling.ex                 # Resampling techniques
│   ├── threshold_optimization.ex     # Threshold optimization
│   ├── adversarial_debiasing.ex      # Adversarial debiasing
│   ├── fair_representation.ex        # Fair representation learning
│   └── calibration.ex                # Calibration techniques
├── report.ex                         # Fairness reporting
└── utils.ex                          # Utility functions
```

## Use Cases

### Loan Approval Models

```elixir
# Ensure fair lending practices
loan_predictions = model_predict(applicant_features)

fairness = ExFairness.fairness_report(
  loan_predictions,
  actual_defaults,
  applicant_race,
  metrics: [:demographic_parity, :equalized_odds]
)

# Apply mitigation if needed
if not fairness.demographic_parity.passes do
  thresholds = ExFairness.Mitigation.optimize_thresholds(
    loan_scores,
    actual_defaults,
    applicant_race
  )
end
```

### Hiring and Recruitment

```elixir
# Analyze resume screening model
screening_results = screen_resumes(resumes)

ExFairness.intersectional_fairness(
  screening_results,
  interview_outcomes,
  sensitive_attrs: [gender, ethnicity],
  attr_names: ["gender", "ethnicity"]
)
```

### Healthcare Risk Prediction

```elixir
# Ensure equitable healthcare predictions
risk_scores = predict_health_risk(patient_data)

ExFairness.calibration(
  risk_scores,
  actual_outcomes,
  patient_race,
  bins: 10
)
```

### Content Moderation

```elixir
# Monitor bias in content moderation
moderation_decisions = moderate_content(posts)

ExFairness.temporal_drift(
  collect_metrics_over_time(moderation_decisions, user_demographics),
  threshold: 0.05
)
```

## Fairness Metrics Reference

### Group Fairness Metrics

| Metric | Definition | When to Use |
|--------|------------|-------------|
| Demographic Parity | P(Y=1\|A=0) = P(Y=1\|A=1) | When equal positive rates across groups is required |
| Equalized Odds | TPR and FPR equal across groups | When both error types matter equally |
| Equal Opportunity | TPR equal across groups | When true positive rate parity is critical |
| Predictive Parity | PPV equal across groups | When precision parity is important |

### Individual Fairness Metrics

| Metric | Definition | When to Use |
|--------|------------|-------------|
| Individual Fairness | Similar individuals get similar predictions | When individual treatment matters |
| Counterfactual Fairness | Predictions unchanged under counterfactuals | When causal fairness is required |

## Fairness-Accuracy Trade-offs

ExFairness helps you navigate the inherent trade-offs between fairness and accuracy:

```elixir
# Analyze the Pareto frontier
tradeoffs = ExFairness.fairness_accuracy_tradeoff(
  model,
  test_data,
  sensitive_attr,
  fairness_metrics: [:demographic_parity, :equalized_odds]
)

# Visualize trade-offs
ExFairness.plot_pareto_frontier(tradeoffs)
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
comprehensive_report = ExFairness.fairness_report(
  predictions,
  labels,
  sensitive_attr,
  metrics: :all
)
```

### 3. Consider Intersectionality

Analyze fairness across multiple sensitive attributes:

```elixir
ExFairness.intersectional_fairness(
  predictions,
  labels,
  sensitive_attrs: [race, gender, age_group]
)
```

### 4. Monitor Over Time

Fairness can degrade as data distributions shift:

```elixir
ExFairness.temporal_drift(metrics_over_time)
```

### 5. Document Trade-offs

Be transparent about fairness-accuracy trade-offs:

```elixir
report = ExFairness.generate_fairness_documentation(
  model,
  fairness_metrics,
  accuracy_metrics
)

File.write!("fairness_report.md", report)
```

## Limitations

- **Impossibility Theorems**: Some fairness definitions are mutually exclusive (e.g., demographic parity and equalized odds with different base rates)
- **Sensitive Attributes**: Requires access to sensitive attributes for measurement
- **Causal Assumptions**: Some metrics require causal assumptions that may not hold
- **Sample Size**: Statistical tests require adequate sample sizes per group

## Research Foundations

### Key Papers

- Hardt, M., et al. (2016). "Equality of Opportunity in Supervised Learning." *NeurIPS*.
- Chouldechova, A. (2017). "Fair prediction with disparate impact." *Big Data*.
- Corbett-Davies, S., & Goel, S. (2018). "The Measure and Mismeasure of Fairness." *Journal of Machine Learning Research*.
- Dwork, C., et al. (2012). "Fairness through awareness." *ITCS*.
- Kusner, M., et al. (2017). "Counterfactual fairness." *NeurIPS*.

### Fairness Frameworks

- **Google's What-If Tool**: Interactive fairness analysis
- **IBM AI Fairness 360**: Comprehensive fairness toolkit (Python)
- **Microsoft Fairlearn**: Fairness assessment and mitigation (Python)
- **Aequitas**: Bias and fairness audit toolkit

## Contributing

This is part of the North Shore AI Research Infrastructure. Contributions are welcome!

## License

MIT License - see [LICENSE](https://github.com/North-Shore-AI/ExFairness/blob/main/LICENSE) file for details
