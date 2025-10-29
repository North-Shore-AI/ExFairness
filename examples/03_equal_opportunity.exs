#!/usr/bin/env elixir

# Equal Opportunity Example
# This example demonstrates how to use the equal opportunity metric
# to ensure equal true positive rates across groups.

IO.puts("\n=== Equal Opportunity Example ===\n")

# Equal opportunity requires that TPR (True Positive Rate) is equal across groups.
# This ensures that qualified individuals from all groups have equal chances
# of receiving positive predictions.
# Formula: P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)

IO.puts("Scenario: College admissions system")
IO.puts("Group 0: Group A, Group 1: Group B")
IO.puts("")

# Example 1: Fair predictions (equal TPR)
IO.puts("Example 1: Fair Predictions")
IO.puts("----------------------------")

predictions = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1])
labels = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.equal_opportunity(predictions, labels, sensitive_attr)

IO.puts("Group A TPR: #{Float.round(result.group_a_tpr * 100, 1)}%")
IO.puts("Group B TPR: #{Float.round(result.group_b_tpr * 100, 1)}%")
IO.puts("Disparity: #{Float.round(result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{result.passes}")
IO.puts("\nInterpretation:")
IO.puts(result.interpretation)

# Example 2: Biased predictions (unequal TPR)
IO.puts("\n\nExample 2: Biased Predictions - Missing Qualified Candidates from Group B")
IO.puts("--------------------------------------------------------------------------")

# Group A: 80% TPR (most qualified candidates admitted)
# Group B: 40% TPR (many qualified candidates rejected)
biased_predictions = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0])
biased_labels = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

biased_result = ExFairness.equal_opportunity(biased_predictions, biased_labels, sensitive_attr)

IO.puts("Group A TPR: #{Float.round(biased_result.group_a_tpr * 100, 1)}%")
IO.puts("Group B TPR: #{Float.round(biased_result.group_b_tpr * 100, 1)}%")
IO.puts("Disparity: #{Float.round(biased_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{biased_result.passes}")
IO.puts("\nInterpretation:")
IO.puts(biased_result.interpretation)

IO.puts("""

This is problematic because:
  - Qualified candidates from Group B are being unfairly rejected
  - The system has lower recall/sensitivity for Group B
  - This creates unequal opportunity despite equal qualifications
""")

# Example 3: When to use Equal Opportunity
IO.puts("\n\nExample 3: When to Use Equal Opportunity")
IO.puts("-----------------------------------------")

IO.puts("""
Use Equal Opportunity when:

1. False Negatives Are More Costly Than False Positives
   - Hiring: Don't want to miss qualified candidates
   - College Admissions: Ensure qualified students get opportunities
   - Loan Approvals: Creditworthy applicants should get loans

2. You Want to Ensure "Equal Access to Opportunity"
   - Qualified individuals should have equal chances
   - Focus is on not disadvantaging any group

3. Different from Equalized Odds:
   - Equal Opportunity: Only cares about TPR (recall)
   - Equalized Odds: Cares about both TPR and FPR
   - Equal Opportunity is less restrictive

Example Use Cases:
  - Job Screening: Don't miss qualified candidates
  - Scholarship Awards: Ensure deserving students aren't overlooked
  - Promotions: Qualified employees should have equal chances
""")

# Example 4: Hiring scenario with larger dataset
IO.puts("\n\nExample 4: Hiring System Audit (1000 applicants)")
IO.puts("-------------------------------------------------")

:rand.seed(:exsplus, {42, 84, 126})

# Simulate hiring decisions
# Group A: 70% of qualified applicants hired (TPR = 0.70)
# Group B: 70% of qualified applicants hired (TPR = 0.70) - FAIR

generate_hiring_data = fn group_size, tpr ->
  # 50% of applicants are qualified
  labels = for _ <- 1..group_size, do: if(:rand.uniform() < 0.5, do: 1, else: 0)

  predictions =
    Enum.map(labels, fn label ->
      case label do
        1 -> if :rand.uniform() < tpr, do: 1, else: 0
        # 10% false positive rate
        0 -> if :rand.uniform() < 0.1, do: 1, else: 0
      end
    end)

  {predictions, labels}
end

{group_a_preds, group_a_labels} = generate_hiring_data.(500, 0.70)
{group_b_preds, group_b_labels} = generate_hiring_data.(500, 0.70)

hire_predictions = Nx.tensor(group_a_preds ++ group_b_preds)
hire_labels = Nx.tensor(group_a_labels ++ group_b_labels)

hire_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {500}),
    Nx.broadcast(1, {500})
  ])

hire_result = ExFairness.equal_opportunity(hire_predictions, hire_labels, hire_sensitive)

IO.puts("Total applicants: 1000 (500 per group)")
IO.puts("")
IO.puts("Group A TPR: #{Float.round(hire_result.group_a_tpr * 100, 1)}%")

IO.puts(
  "  (Of qualified Group A applicants, #{Float.round(hire_result.group_a_tpr * 100, 1)}% were hired)"
)

IO.puts("")
IO.puts("Group B TPR: #{Float.round(hire_result.group_b_tpr * 100, 1)}%")

IO.puts(
  "  (Of qualified Group B applicants, #{Float.round(hire_result.group_b_tpr * 100, 1)}% were hired)"
)

IO.puts("")
IO.puts("Disparity: #{Float.round(hire_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{hire_result.passes}")

IO.puts("""

This hiring system is FAIR because:
  - Qualified candidates from both groups have equal chances
  - The system doesn't systematically miss qualified candidates from either group
  - Equal opportunity is achieved
""")

# Example 5: Biased hiring system
IO.puts("\n\nExample 5: Biased Hiring System")
IO.puts("--------------------------------")

{group_a_preds_biased, group_a_labels_biased} = generate_hiring_data.(500, 0.80)
{group_b_preds_biased, group_b_labels_biased} = generate_hiring_data.(500, 0.50)

biased_hire_predictions = Nx.tensor(group_a_preds_biased ++ group_b_preds_biased)
biased_hire_labels = Nx.tensor(group_a_labels_biased ++ group_b_labels_biased)

biased_hire_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {500}),
    Nx.broadcast(1, {500})
  ])

biased_hire_result =
  ExFairness.equal_opportunity(biased_hire_predictions, biased_hire_labels, biased_hire_sensitive)

IO.puts("Group A TPR: #{Float.round(biased_hire_result.group_a_tpr * 100, 1)}%")
IO.puts("Group B TPR: #{Float.round(biased_hire_result.group_b_tpr * 100, 1)}%")
IO.puts("Disparity: #{Float.round(biased_hire_result.disparity * 100, 1)}%")
IO.puts("Passes fairness test: #{biased_hire_result.passes}")

IO.puts("""

This hiring system is BIASED because:
  - Qualified Group B candidates have only 50% chance of being hired
  - Qualified Group A candidates have 80% chance of being hired
  - The system systematically disadvantages qualified Group B candidates
  - This violates equal opportunity
""")

IO.puts("\n=== Example Complete ===\n")
