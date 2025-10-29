#!/usr/bin/env elixir

# Predictive Parity Example
# This example demonstrates how to use the predictive parity metric
# to ensure equal precision across groups.

IO.puts("\n=== Predictive Parity Example ===\n")

# Predictive parity requires that PPV (Positive Predictive Value / Precision)
# is equal across groups. This ensures that a positive prediction has the same
# meaning regardless of group membership.
# Formula: P(Y = 1 | Ŷ = 1, A = 0) = P(Y = 1 | Ŷ = 1, A = 1)
# Or: PPV_A = PPV_B where PPV = TP / (TP + FP)

IO.puts("Scenario: Credit risk assessment")
IO.puts("Group 0: Group A, Group 1: Group B")
IO.puts("")

# Example 1: Fair predictions (equal PPV)
IO.puts("Example 1: Fair Predictions")
IO.puts("----------------------------")

predictions = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1])
labels = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.predictive_parity(predictions, labels, sensitive_attr)

IO.puts("Group A PPV (Precision): #{Float.round(result.group_a_ppv * 100, 1)}%")
IO.puts("Group B PPV (Precision): #{Float.round(result.group_b_ppv * 100, 1)}%")
IO.puts("Disparity: #{Float.round(result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{result.passes}")
IO.puts("\nInterpretation:")
IO.puts(result.interpretation)

# Example 2: Biased predictions (unequal PPV)
IO.puts("\n\nExample 2: Biased Predictions - Different Precision Across Groups")
IO.puts("------------------------------------------------------------------")

# Group A: High precision (80% of approved actually don't default)
# Group B: Lower precision (50% of approved actually don't default)
biased_predictions = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
biased_labels = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

biased_result = ExFairness.predictive_parity(biased_predictions, biased_labels, sensitive_attr)

IO.puts("Group A PPV: #{Float.round(biased_result.group_a_ppv * 100, 1)}%")
IO.puts("Group B PPV: #{Float.round(biased_result.group_b_ppv * 100, 1)}%")
IO.puts("Disparity: #{Float.round(biased_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{biased_result.passes}")
IO.puts("\nInterpretation:")
IO.puts(biased_result.interpretation)

IO.puts("""

This is problematic because:
  - "Approved" means different default risks for different groups
  - Group B approvals have higher false positive rate
  - The prediction doesn't have consistent meaning across groups
""")

# Example 3: When to use Predictive Parity
IO.puts("\n\nExample 3: When to Use Predictive Parity")
IO.puts("-----------------------------------------")

IO.puts("""
Use Predictive Parity when:

1. Positive Predictions Should Mean the Same Thing
   - Credit: "Approved" should mean similar default risk
   - Risk Assessment: "High risk" should mean similar actual risk
   - Medical: "Disease detected" should mean similar true disease rate

2. Users Make Decisions Based on Predictions
   - Lenders decide interest rates based on approval
   - Judges make bail decisions based on risk scores
   - Doctors order treatments based on diagnosis

3. Calibration Matters
   - You want predictions to be "well-calibrated" across groups
   - A score of 0.8 should mean 80% probability for all groups

Example Use Cases:
  - Credit Scoring: Equal default rates for approved loans
  - Insurance Pricing: Equal claim rates for same premium
  - Medical Diagnosis: Equal disease prevalence for positive tests
  - Recidivism Prediction: Equal re-offense rates for same risk level
""")

# Example 4: Credit scoring scenario
IO.puts("\n\nExample 4: Credit Scoring System (1000 applications)")
IO.puts("-----------------------------------------------------")

:rand.seed(:exsplus, {100, 200, 300})

# Simulate credit decisions
# Fair system: Both groups have 85% PPV (precision)
generate_credit_data = fn group_size, ppv ->
  # Generate predictions: 40% approved, 60% rejected
  predictions = for _ <- 1..group_size, do: if(:rand.uniform() < 0.4, do: 1, else: 0)

  # For each positive prediction, assign true label based on PPV
  labels =
    Enum.map(predictions, fn pred ->
      case pred do
        # PPV: of approved, this % are good
        1 -> if :rand.uniform() < ppv, do: 1, else: 0
        # Most rejected would have defaulted
        0 -> if :rand.uniform() < 0.8, do: 0, else: 1
      end
    end)

  {predictions, labels}
end

{group_a_preds, group_a_labels} = generate_credit_data.(500, 0.85)
{group_b_preds, group_b_labels} = generate_credit_data.(500, 0.85)

credit_predictions = Nx.tensor(group_a_preds ++ group_b_preds)
credit_labels = Nx.tensor(group_a_labels ++ group_b_labels)

credit_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {500}),
    Nx.broadcast(1, {500})
  ])

credit_result = ExFairness.predictive_parity(credit_predictions, credit_labels, credit_sensitive)

IO.puts("Total applications: 1000 (500 per group)")
IO.puts("")
IO.puts("Group A PPV: #{Float.round(credit_result.group_a_ppv * 100, 1)}%")

IO.puts(
  "  (Of approved Group A loans, #{Float.round(credit_result.group_a_ppv * 100, 1)}% will be repaid)"
)

IO.puts("")
IO.puts("Group B PPV: #{Float.round(credit_result.group_b_ppv * 100, 1)}%")

IO.puts(
  "  (Of approved Group B loans, #{Float.round(credit_result.group_b_ppv * 100, 1)}% will be repaid)"
)

IO.puts("")
IO.puts("Disparity: #{Float.round(credit_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{credit_result.passes}")

IO.puts("""

This credit system is FAIR because:
  - "Approved" means similar repayment likelihood for both groups
  - Lenders can trust the prediction equally for both groups
  - Risk is calibrated: approval has same meaning
""")

# Example 5: Biased credit system
IO.puts("\n\nExample 5: Biased Credit System")
IO.puts("--------------------------------")

{group_a_preds_biased, group_a_labels_biased} = generate_credit_data.(500, 0.90)
{group_b_preds_biased, group_b_labels_biased} = generate_credit_data.(500, 0.60)

biased_credit_predictions = Nx.tensor(group_a_preds_biased ++ group_b_preds_biased)
biased_credit_labels = Nx.tensor(group_a_labels_biased ++ group_b_labels_biased)

biased_credit_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {500}),
    Nx.broadcast(1, {500})
  ])

biased_credit_result =
  ExFairness.predictive_parity(
    biased_credit_predictions,
    biased_credit_labels,
    biased_credit_sensitive
  )

IO.puts("Group A PPV: #{Float.round(biased_credit_result.group_a_ppv * 100, 1)}%")
IO.puts("  (90% of approved Group A loans will be repaid)")
IO.puts("")
IO.puts("Group B PPV: #{Float.round(biased_credit_result.group_b_ppv * 100, 1)}%")
IO.puts("  (Only 60% of approved Group B loans will be repaid)")
IO.puts("")
IO.puts("Disparity: #{Float.round(biased_credit_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{biased_credit_result.passes}")

IO.puts("""

This credit system is BIASED because:
  - "Approved" means very different things for the two groups
  - Group B approvals have 40% default rate vs 10% for Group A
  - Group B may be charged higher interest rates unfairly
  - The prediction is not calibrated across groups
""")

# Example 6: Understanding the trade-off
IO.puts("\n\nExample 6: The Calibration-Fairness Trade-off")
IO.puts("----------------------------------------------")

IO.puts("""
Important Note: Predictive Parity vs Equal Opportunity

These metrics can CONFLICT when base rates differ:
  - If groups have different actual positive rates (base rates)
  - You CANNOT satisfy both Predictive Parity and Equal Opportunity
  - This is a proven mathematical impossibility (Chouldechova 2017)

Example:
  - Group A: 30% actually default on loans
  - Group B: 10% actually default on loans

  A perfect predictor would:
    ✓ Satisfy Equal Opportunity (TPR = 100% for both)
    ✓ Satisfy Predictive Parity (PPV = 100% for both)
    ✗ VIOLATE Demographic Parity (30% vs 10% approval rates)

  You must choose which fairness notion matters most for your use case:
    - Predictive Parity: When predictions guide decisions (lending, pricing)
    - Equal Opportunity: When access to opportunities matters (hiring, admissions)
    - Demographic Parity: When representation matters (advertising, recommendations)
""")

IO.puts("\n=== Example Complete ===\n")
