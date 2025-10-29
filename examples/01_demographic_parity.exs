#!/usr/bin/env elixir

# Demographic Parity Example
# This example demonstrates how to use the demographic parity metric
# to detect bias in binary classification predictions.

IO.puts("\n=== Demographic Parity Example ===\n")

# Demographic parity requires that the probability of a positive prediction
# is equal across groups defined by sensitive attributes (e.g., race, gender).
# Formula: P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)

IO.puts("Scenario: Loan approval system")
IO.puts("Group 0: Group A, Group 1: Group B")
IO.puts("")

# Example 1: Fair predictions (equal positive rates)
IO.puts("Example 1: Fair Predictions")
IO.puts("----------------------------")

predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.demographic_parity(predictions, sensitive_attr)

IO.puts("Group A positive rate: #{Float.round(result.group_a_rate * 100, 1)}%")
IO.puts("Group B positive rate: #{Float.round(result.group_b_rate * 100, 1)}%")
IO.puts("Disparity: #{Float.round(result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{result.passes}")
IO.puts("\nInterpretation:")
IO.puts(result.interpretation)

# Example 2: Biased predictions (unequal positive rates)
IO.puts("\n\nExample 2: Biased Predictions")
IO.puts("------------------------------")

biased_predictions = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

biased_result = ExFairness.demographic_parity(biased_predictions, sensitive_attr)

IO.puts("Group A positive rate: #{Float.round(biased_result.group_a_rate * 100, 1)}%")
IO.puts("Group B positive rate: #{Float.round(biased_result.group_b_rate * 100, 1)}%")
IO.puts("Disparity: #{Float.round(biased_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{biased_result.passes}")
IO.puts("\nInterpretation:")
IO.puts(biased_result.interpretation)

# Example 3: Custom threshold
IO.puts("\n\nExample 3: Custom Fairness Threshold")
IO.puts("-------------------------------------")

IO.puts("Using stricter threshold of 5% instead of default 10%")
strict_result = ExFairness.demographic_parity(predictions, sensitive_attr, threshold: 0.05)

IO.puts("Group A positive rate: #{Float.round(strict_result.group_a_rate * 100, 1)}%")
IO.puts("Group B positive rate: #{Float.round(strict_result.group_b_rate * 100, 1)}%")
IO.puts("Disparity: #{Float.round(strict_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test (5% threshold): #{strict_result.passes}")

# Example 4: Real-world simulation with larger dataset
IO.puts("\n\nExample 4: Real-World Simulation (1000 samples)")
IO.puts("------------------------------------------------")

# Simulate a hiring model with slight bias
# Group A: 60% approval rate, Group B: 45% approval rate
:rand.seed(:exsplus, {42, 42, 42})

group_a_predictions =
  for _ <- 1..500, do: if(:rand.uniform() < 0.60, do: 1, else: 0)

group_b_predictions =
  for _ <- 1..500, do: if(:rand.uniform() < 0.45, do: 1, else: 0)

large_predictions = Nx.tensor(group_a_predictions ++ group_b_predictions)

large_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {500}),
    Nx.broadcast(1, {500})
  ])

large_result = ExFairness.demographic_parity(large_predictions, large_sensitive)

IO.puts("Dataset size: 1000 (500 per group)")
IO.puts("Group A positive rate: #{Float.round(large_result.group_a_rate * 100, 1)}%")
IO.puts("Group B positive rate: #{Float.round(large_result.group_b_rate * 100, 1)}%")
IO.puts("Disparity: #{Float.round(large_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{large_result.passes}")

IO.puts("\n=== Example Complete ===\n")
