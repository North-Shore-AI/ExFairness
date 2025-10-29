#!/usr/bin/env elixir

# Equalized Odds Example
# This example demonstrates how to use the equalized odds metric
# to ensure equal error rates across groups.

IO.puts("\n=== Equalized Odds Example ===\n")

# Equalized odds requires that both TPR (True Positive Rate) and FPR (False Positive Rate)
# are equal across groups. This ensures that the model's error rates are the same
# regardless of group membership.
# Formula: P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1) [Equal TPR]
#          P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1) [Equal FPR]

IO.puts("Scenario: Medical diagnosis system")
IO.puts("Group 0: Group A, Group 1: Group B")
IO.puts("")

# Example 1: Fair predictions (equal TPR and FPR)
IO.puts("Example 1: Fair Predictions")
IO.puts("----------------------------")

predictions = Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.equalized_odds(predictions, labels, sensitive_attr)

IO.puts("Group A TPR: #{Float.round(result.group_a_tpr * 100, 1)}%")
IO.puts("Group B TPR: #{Float.round(result.group_b_tpr * 100, 1)}%")
IO.puts("TPR Disparity: #{Float.round(result.tpr_disparity * 100, 1)}%")
IO.puts("")
IO.puts("Group A FPR: #{Float.round(result.group_a_fpr * 100, 1)}%")
IO.puts("Group B FPR: #{Float.round(result.group_b_fpr * 100, 1)}%")
IO.puts("FPR Disparity: #{Float.round(result.fpr_disparity * 100, 1)}%")
IO.puts("")
IO.puts("Passes fairness test: #{result.passes}")
IO.puts("\nInterpretation:")
IO.puts(result.interpretation)

# Example 2: Biased predictions (unequal TPR)
IO.puts("\n\nExample 2: Biased Predictions - Higher False Negatives for Group B")
IO.puts("-------------------------------------------------------------------")

# Group A: Good recall (catches most positive cases)
# Group B: Poor recall (misses many positive cases)
biased_predictions = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1])
biased_labels = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

biased_result = ExFairness.equalized_odds(biased_predictions, biased_labels, sensitive_attr)

IO.puts("Group A TPR: #{Float.round(biased_result.group_a_tpr * 100, 1)}%")
IO.puts("Group B TPR: #{Float.round(biased_result.group_b_tpr * 100, 1)}%")
IO.puts("TPR Disparity: #{Float.round(biased_result.tpr_disparity * 100, 1)}%")
IO.puts("")
IO.puts("Group A FPR: #{Float.round(biased_result.group_a_fpr * 100, 1)}%")
IO.puts("Group B FPR: #{Float.round(biased_result.group_b_fpr * 100, 1)}%")
IO.puts("FPR Disparity: #{Float.round(biased_result.fpr_disparity * 100, 1)}%")
IO.puts("")
IO.puts("Passes fairness test: #{biased_result.passes}")
IO.puts("\nInterpretation:")
IO.puts(biased_result.interpretation)

# Example 3: Understanding the metrics
IO.puts("\n\nExample 3: Understanding TPR and FPR")
IO.puts("-------------------------------------")

IO.puts("""
TPR (True Positive Rate / Recall / Sensitivity):
  - Measures: Of all actual positive cases, how many did we correctly identify?
  - Formula: TP / (TP + FN)
  - High TPR means: Few false negatives (not missing positive cases)

FPR (False Positive Rate):
  - Measures: Of all actual negative cases, how many did we incorrectly flag as positive?
  - Formula: FP / (FP + TN)
  - Low FPR means: Few false alarms

For medical diagnosis:
  - High TPR: Good at catching disease (few missed diagnoses)
  - Low FPR: Good at avoiding false alarms (few unnecessary treatments)

Equalized odds ensures BOTH error types are equal across groups.
""")

# Example 4: Criminal justice scenario
IO.puts("\n\nExample 4: Criminal Justice Scenario")
IO.puts("-------------------------------------")

# Simulating a risk assessment tool
# Equal error rates are critical in criminal justice
:rand.seed(:exsplus, {123, 456, 789})

# Generate synthetic data with balanced error rates
generate_predictions = fn group_size, tpr, fpr ->
  # Generate 60% positive, 40% negative
  labels = for _ <- 1..group_size, do: if(:rand.uniform() < 0.6, do: 1, else: 0)

  predictions =
    Enum.map(labels, fn label ->
      case label do
        # TPR: correctly predict positive
        1 -> if :rand.uniform() < tpr, do: 1, else: 0
        # FPR: incorrectly predict positive
        0 -> if :rand.uniform() < fpr, do: 1, else: 0
      end
    end)

  {predictions, labels}
end

# Both groups have same error rates (fair)
{group_a_preds, group_a_labels} = generate_predictions.(250, 0.85, 0.15)
{group_b_preds, group_b_labels} = generate_predictions.(250, 0.85, 0.15)

cj_predictions = Nx.tensor(group_a_preds ++ group_b_preds)
cj_labels = Nx.tensor(group_a_labels ++ group_b_labels)

cj_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {250}),
    Nx.broadcast(1, {250})
  ])

cj_result = ExFairness.equalized_odds(cj_predictions, cj_labels, cj_sensitive)

IO.puts("Risk Assessment Tool (500 cases)")
IO.puts("")
IO.puts("Group A TPR: #{Float.round(cj_result.group_a_tpr * 100, 1)}%")
IO.puts("Group B TPR: #{Float.round(cj_result.group_b_tpr * 100, 1)}%")
IO.puts("TPR Disparity: #{Float.round(cj_result.tpr_disparity * 100, 1)}%")
IO.puts("")
IO.puts("Group A FPR: #{Float.round(cj_result.group_a_fpr * 100, 1)}%")
IO.puts("Group B FPR: #{Float.round(cj_result.group_b_fpr * 100, 1)}%")
IO.puts("FPR Disparity: #{Float.round(cj_result.fpr_disparity * 100, 1)}%")
IO.puts("")
IO.puts("Passes fairness test: #{cj_result.passes}")

IO.puts("\n=== Example Complete ===\n")
