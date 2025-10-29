#!/usr/bin/env elixir

# Bias Mitigation with Reweighting Example
# This example demonstrates how to use reweighting as a pre-processing
# technique to achieve fairness.

IO.puts("\n=== Bias Mitigation with Reweighting ===\n")

# Reweighting is a pre-processing technique that assigns different weights
# to training samples to achieve fairness. Samples from underrepresented
# group-label combinations receive higher weights.
#
# Mathematical Foundation:
# For demographic parity: w(a, y) = P(Y = y) / P(A = a, Y = y)
# This ensures all group-label combinations have equal expected weight.

IO.puts("Technique: Reweighting (Pre-processing)")
IO.puts("Goal: Adjust sample weights to balance representation")
IO.puts("")

# Example 1: Basic reweighting for demographic parity
IO.puts("Example 1: Reweighting for Demographic Parity")
IO.puts("-----------------------------------------------")

# Training data with imbalanced groups
labels = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

IO.puts("Original data distribution:")
IO.puts("  Group A (n=10): 4 positive, 6 negative (40% positive rate)")
IO.puts("  Group B (n=10): 2 positive, 8 negative (20% positive rate)")
IO.puts("  Overall imbalance: Group A has 2x positive rate")

# Compute reweighting for demographic parity
weights =
  ExFairness.Mitigation.Reweighting.compute_weights(
    labels,
    sensitive_attr,
    target: :demographic_parity
  )

IO.puts("\nReweighting strategy:")
IO.puts("  Samples from (Group B, Positive) get higher weights")
IO.puts("  Samples from (Group A, Negative) get higher weights")
IO.puts("  This balances representation across all (group, label) combinations")

IO.puts("\nComputed weights (sample):")
weights_list = Nx.to_flat_list(weights)
labels_list = Nx.to_flat_list(labels)
sensitive_list = Nx.to_flat_list(sensitive_attr)

Enum.zip([Enum.take(weights_list, 6), Enum.take(labels_list, 6), Enum.take(sensitive_list, 6)])
|> Enum.with_index()
|> Enum.each(fn {{weight, label, group}, idx} ->
  IO.puts("  Sample #{idx}: Group=#{group}, Label=#{label}, Weight=#{Float.round(weight, 3)}")
end)

mean_weight = Nx.mean(weights) |> Nx.to_number()
IO.puts("\nMean weight: #{Float.round(mean_weight, 3)} (normalized to 1.0)")

# Example 2: Reweighting for equalized odds
IO.puts("\n\nExample 2: Reweighting for Equalized Odds")
IO.puts("------------------------------------------")

# More complex reweighting that balances all confusion matrix cells
eo_weights =
  ExFairness.Mitigation.Reweighting.compute_weights(
    labels,
    sensitive_attr,
    target: :equalized_odds
  )

IO.puts("Equalized odds reweighting:")
IO.puts("  Balances all four confusion matrix combinations:")
IO.puts("    - (Group A, True Positive)")
IO.puts("    - (Group A, False Positive)")
IO.puts("    - (Group B, True Positive)")
IO.puts("    - (Group B, False Positive)")

eo_weights_list = Nx.to_flat_list(eo_weights)
IO.puts("\nComputed weights (sample):")

Enum.zip([Enum.take(eo_weights_list, 6), Enum.take(labels_list, 6), Enum.take(sensitive_list, 6)])
|> Enum.with_index()
|> Enum.each(fn {{weight, label, group}, idx} ->
  IO.puts("  Sample #{idx}: Group=#{group}, Label=#{label}, Weight=#{Float.round(weight, 3)}")
end)

# Example 3: Simulating training with weights
IO.puts("\n\nExample 3: Impact of Reweighting on Model Training")
IO.puts("---------------------------------------------------")

IO.puts("""
How to use weights in training:

1. With Loss Functions:
   loss_fn = fn pred, label, weight ->
     weight * binary_cross_entropy(pred, label)
   end

2. With Sample Replication:
   - Round weights to integers
   - Replicate samples according to weights
   - Standard training on replicated dataset

3. With ML Libraries:
   # Scholar (when supported):
   model = Scholar.Linear.LogisticRegression.fit(
     features, labels,
     sample_weights: weights
   )

   # Custom training loop:
   weighted_gradient = gradient * weight
""")

# Example 4: Complete fairness improvement workflow
IO.puts("\n\nExample 4: Complete Fairness Improvement Workflow")
IO.puts("--------------------------------------------------")

:rand.seed(:exsplus, {42, 84, 126})

# Step 1: Generate biased training data
IO.puts("\nStep 1: Assess Initial Bias")
IO.puts("---------------------------")

generate_biased_data = fn group_size, pos_rate ->
  labels = for _ <- 1..group_size, do: if(:rand.uniform() < pos_rate, do: 1, else: 0)
  # Simulate predictions based on labels with some noise
  predictions =
    Enum.map(labels, fn label ->
      if :rand.uniform() < 0.8 do
        label
      else
        1 - label
      end
    end)

  {predictions, labels}
end

# Group A: 60% positive rate
# Group B: 30% positive rate
{group_a_preds, group_a_labels} = generate_biased_data.(100, 0.60)
{group_b_preds, group_b_labels} = generate_biased_data.(100, 0.30)

initial_predictions = Nx.tensor(group_a_preds ++ group_b_preds)
initial_labels = Nx.tensor(group_a_labels ++ group_b_labels)

initial_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {100}),
    Nx.broadcast(1, {100})
  ])

# Check initial fairness
initial_report =
  ExFairness.fairness_report(
    initial_predictions,
    initial_labels,
    initial_sensitive
  )

IO.puts(
  "Initial model fairness: #{initial_report.passed_count}/#{initial_report.total_count} metrics passed"
)

IO.puts(
  "Demographic parity disparity: #{Float.round(initial_report.demographic_parity.disparity * 100, 1)}%"
)

# Step 2: Compute reweighting
IO.puts("\nStep 2: Compute Fairness Weights")
IO.puts("--------------------------------")

fairness_weights =
  ExFairness.Mitigation.Reweighting.compute_weights(
    initial_labels,
    initial_sensitive,
    target: :demographic_parity
  )

IO.puts("Weights computed successfully")
IO.puts("Mean weight: #{Float.round(Nx.mean(fairness_weights) |> Nx.to_number(), 3)}")

# Count samples by group and label to show reweighting effect
group_a_pos = Enum.count(Enum.zip(group_a_labels, []), fn {l, _} -> l == 1 end)
group_a_neg = 100 - group_a_pos
group_b_pos = Enum.count(Enum.zip(group_b_labels, []), fn {l, _} -> l == 1 end)
group_b_neg = 100 - group_b_pos

IO.puts("\nOriginal sample counts:")
IO.puts("  Group A: #{group_a_pos} positive, #{group_a_neg} negative")
IO.puts("  Group B: #{group_b_pos} positive, #{group_b_neg} negative")

# Step 3: Simulate retraining (conceptual)
IO.puts("\nStep 3: Retrain Model with Weights")
IO.puts("-----------------------------------")
IO.puts("(Conceptual - in practice, use weights in your training algorithm)")

# Simulate improved predictions with better balance
{group_a_improved, group_a_labels_improved} = generate_biased_data.(100, 0.45)
{group_b_improved, group_b_labels_improved} = generate_biased_data.(100, 0.45)

improved_predictions = Nx.tensor(group_a_improved ++ group_b_improved)
improved_labels = Nx.tensor(group_a_labels_improved ++ group_b_labels_improved)

improved_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {100}),
    Nx.broadcast(1, {100})
  ])

# Step 4: Validate improvement
IO.puts("\nStep 4: Validate Fairness Improvement")
IO.puts("--------------------------------------")

improved_report =
  ExFairness.fairness_report(
    improved_predictions,
    improved_labels,
    improved_sensitive
  )

IO.puts(
  "Improved model fairness: #{improved_report.passed_count}/#{improved_report.total_count} metrics passed"
)

IO.puts(
  "Demographic parity disparity: #{Float.round(improved_report.demographic_parity.disparity * 100, 1)}%"
)

IO.puts("\nImprovement Summary:")
IO.puts("  Before: #{initial_report.passed_count}/#{initial_report.total_count} metrics passed")
IO.puts("  After:  #{improved_report.passed_count}/#{improved_report.total_count} metrics passed")

IO.puts(
  "  Gain:   #{improved_report.passed_count - initial_report.passed_count} additional metrics passed"
)

# Example 5: Understanding weight computation
IO.puts("\n\nExample 5: Understanding Weight Computation")
IO.puts("--------------------------------------------")

IO.puts("""
How reweighting works:

For Demographic Parity:
  1. Count samples in each (group, label) combination:
     - N(A=0, Y=0), N(A=0, Y=1)
     - N(A=1, Y=0), N(A=1, Y=1)

  2. Compute joint probabilities:
     - P(A=0, Y=0) = N(A=0, Y=0) / N_total
     - P(A=0, Y=1) = N(A=0, Y=1) / N_total
     - Similar for A=1

  3. Compute marginal probabilities:
     - P(Y=0) = P(A=0, Y=0) + P(A=1, Y=0)
     - P(Y=1) = P(A=0, Y=1) + P(A=1, Y=1)

  4. Assign weights:
     - w(A=a, Y=y) = P(Y=y) / P(A=a, Y=y)
     - This makes all (A, Y) combinations equally weighted

  5. Normalize:
     - Scale weights so mean = 1.0
     - Maintains total sample weight

Example with your data:
  Group A: 4 pos, 6 neg (total: 10)
  Group B: 2 pos, 8 neg (total: 10)

  Joint probabilities:
    P(A=0, Y=1) = 4/20 = 0.20
    P(A=0, Y=0) = 6/20 = 0.30
    P(A=1, Y=1) = 2/20 = 0.10
    P(A=1, Y=0) = 8/20 = 0.40

  Marginal probabilities:
    P(Y=1) = 0.20 + 0.10 = 0.30
    P(Y=0) = 0.30 + 0.40 = 0.70

  Weights before normalization:
    w(A=0, Y=1) = 0.30 / 0.20 = 1.50
    w(A=0, Y=0) = 0.70 / 0.30 = 2.33
    w(A=1, Y=1) = 0.30 / 0.10 = 3.00  (highest - underrepresented)
    w(A=1, Y=0) = 0.70 / 0.40 = 1.75

  After normalization, mean weight = 1.0
""")

# Example 6: When to use reweighting
IO.puts("\n\nExample 6: When to Use Reweighting")
IO.puts("-----------------------------------")

IO.puts("""
Advantages:
  ✓ Simple to implement
  ✓ Works with any ML algorithm that supports sample weights
  ✓ Preserves all training data
  ✓ Theoretically grounded
  ✓ Can target specific fairness metrics

Disadvantages:
  ✗ May slightly reduce overall accuracy
  ✗ Requires algorithm support for sample weights
  ✗ Extreme weights can cause numerical instability
  ✗ Doesn't change features or labels

Best for:
  - Imbalanced training data
  - When you can modify training procedure
  - When demographic parity or equalized odds is the goal
  - Pre-processing before model training

Not ideal for:
  - Post-hoc fairness (use threshold optimization instead)
  - When accuracy is paramount
  - Very small datasets (weights may be unstable)
  - When algorithm doesn't support sample weights
""")

# Example 7: Monitoring weight distribution
IO.puts("\n\nExample 7: Monitoring Weight Distribution")
IO.puts("------------------------------------------")

# Analyze weight distribution
weights_list_full = Nx.to_flat_list(weights)
min_weight = Enum.min(weights_list_full)
max_weight = Enum.max(weights_list_full)
mean_weight_full = Enum.sum(weights_list_full) / length(weights_list_full)

IO.puts("Weight statistics:")
IO.puts("  Min weight: #{Float.round(min_weight, 3)}")
IO.puts("  Max weight: #{Float.round(max_weight, 3)}")
IO.puts("  Mean weight: #{Float.round(mean_weight_full, 3)}")
IO.puts("  Weight range: #{Float.round(max_weight / min_weight, 2)}x")

if max_weight / min_weight > 10 do
  IO.puts("\n⚠ Warning: Large weight range detected")
  IO.puts("  Consider:")
  IO.puts("    - Collecting more data from underrepresented groups")
  IO.puts("    - Using weight clipping to limit extreme weights")
  IO.puts("    - Alternative fairness techniques (resampling, etc.)")
else
  IO.puts("\n✓ Weight range is reasonable")
end

IO.puts("\n=== Example Complete ===\n")
