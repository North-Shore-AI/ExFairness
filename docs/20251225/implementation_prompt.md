# ExFairness - Implementation Prompt

**Date:** 2025-12-25
**Target:** Add Crucible.Stage wrapper (ExFairness.Stage implementing Crucible.Stage behaviour)

## Required Reading

Before implementing, read these files in order:

### 1. Core Reference (Crucible Framework)

```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage.ex
```
Lines 1-18 - The Crucible.Stage behaviour definition:
- `@callback run(context :: Context.t(), opts :: opts()) :: {:ok, Context.t()} | {:error, term()}`
- `@callback describe(opts :: opts()) :: map()` (optional)

```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/fairness.ex
```
Lines 1-339 - Reference implementation of a fairness stage in crucible_framework:
- Lines 81-111: `@behaviour Crucible.Stage`, `run/2`, `describe/1`
- Lines 118-162: Main evaluation logic pattern
- Lines 227-252: Data extraction from context

### 2. ExFairness Current Implementation

```
/home/home/p/g/North-Shore-AI/ExFairness/lib/ex_fairness.ex
```
Lines 1-319 - Main facade with all metric functions:
- Lines 73-75: `demographic_parity/3`
- Lines 106-108: `equalized_odds/4`
- Lines 129-131: `equal_opportunity/4`
- Lines 152-154: `predictive_parity/4`
- Lines 182-184: `calibration/4`
- Lines 212-214: `fairness_report/4`
- Lines 261-318: `evaluate/5` for CrucibleIR integration

```
/home/home/p/g/North-Shore-AI/ExFairness/lib/ex_fairness/stage.ex
```
Lines 1-311 - Current ad-hoc stage (to be replaced/refactored):
- Lines 127-152: Current `run/2` implementation
- Lines 172-174: Current `describe/1`
- Lines 178-216: Tensor extraction from outputs
- Lines 219-232: Metric computation
- Lines 235-272: Individual metric handlers

```
/home/home/p/g/North-Shore-AI/ExFairness/lib/ex_fairness/report.ex
```
Lines 86-109: Report generation for metrics

### 3. Utility Modules

```
/home/home/p/g/North-Shore-AI/ExFairness/lib/ex_fairness/validation.ex
```
Lines 1-240 - Input validation:
- Lines 33-39: `validate_predictions!/1`
- Lines 65-71: `validate_labels!/1`
- Lines 99-109: `validate_sensitive_attr!/2`
- Lines 137-155: `validate_matching_shapes!/2`

### 4. Tests

```
/home/home/p/g/North-Shore-AI/ExFairness/test/ex_fairness/stage_test.exs
```
Lines 1-421 - Current stage tests (need updating for new implementation)

## Current Module Structure

### Fairness Metrics

| Module | File | Key Function | Lines |
|--------|------|--------------|-------|
| DemographicParity | `lib/ex_fairness/metrics/demographic_parity.ex` | `compute/3` | 96-134 |
| EqualizedOdds | `lib/ex_fairness/metrics/equalized_odds.ex` | `compute/4` | 102-158 |
| EqualOpportunity | `lib/ex_fairness/metrics/equal_opportunity.ex` | `compute/4` | 94-134 |
| PredictiveParity | `lib/ex_fairness/metrics/predictive_parity.ex` | `compute/4` | 93-133 |
| Calibration | `lib/ex_fairness/metrics/calibration.ex` | `compute/4` | 119-172 |

### Detection & Mitigation

| Module | File | Key Function | Lines |
|--------|------|--------------|-------|
| DisparateImpact | `lib/ex_fairness/detection/disparate_impact.ex` | `detect/3` | 92-126 |
| Reweighting | `lib/ex_fairness/mitigation/reweighting.ex` | `compute_weights/3` | 74-90 |

### Utilities

| Module | File | Key Functions | Lines |
|--------|------|---------------|-------|
| Utils | `lib/ex_fairness/utils.ex` | `group_positive_rates/2` | 118-126 |
| Utils.Metrics | `lib/ex_fairness/utils/metrics.ex` | `confusion_matrix/3`, `true_positive_rate/3` | 46-60, 88-94 |
| Utils.Bootstrap | `lib/ex_fairness/utils/bootstrap.ex` | `confidence_interval/3` | 108-145 |
| Utils.StatisticalTests | `lib/ex_fairness/utils/statistical_tests.ex` | `two_proportion_test/3` | 100-154 |

## Implementation Requirements

### Task: Create ExFairness.CrucibleStage

Create a new module that properly implements the `Crucible.Stage` behaviour from crucible_framework.

### File to Create

```
/home/home/p/g/North-Shore-AI/ExFairness/lib/ex_fairness/crucible_stage.ex
```

### Behaviour Contract

```elixir
defmodule ExFairness.CrucibleStage do
  @moduledoc """
  Crucible.Stage implementation for fairness evaluation.

  This stage integrates ExFairness into crucible_framework pipelines,
  providing fairness metric evaluation on model outputs.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context

  @impl true
  @spec run(Context.t(), map()) :: {:ok, Context.t()} | {:error, term()}
  def run(context, opts) do
    # Implementation
  end

  @impl true
  @spec describe(map()) :: map()
  def describe(opts) do
    # Return map with stage metadata
  end
end
```

### Context Requirements

The stage should:

1. **Get data from context:**
   - `context.outputs` - List of maps with predictions/labels/sensitive attributes
   - `context.experiment.reliability.fairness` - CrucibleIR.Reliability.Fairness config
   - OR `context.assigns[:fairness_*]` - Pre-computed tensors

2. **Extract tensors:**
   - predictions: From outputs or assigns
   - labels: From outputs or assigns
   - sensitive_attr: Based on `config.group_by` field
   - probabilities: Optional, for calibration

3. **Run fairness evaluation:**
   - Compute each metric specified in config.metrics
   - Use config.threshold for pass/fail determination
   - Support all 5 metrics: demographic_parity, equalized_odds, equal_opportunity, predictive_parity, calibration

4. **Merge results into context:**
   - Store in `context.metrics.fairness`
   - Include: metrics map, overall_passes, violations list

### TDD Approach

Write tests FIRST in this order:

#### Step 1: Create test file

```
/home/home/p/g/North-Shore-AI/ExFairness/test/ex_fairness/crucible_stage_test.exs
```

#### Step 2: Test cases to implement

```elixir
defmodule ExFairness.CrucibleStageTest do
  use ExUnit.Case, async: true

  alias ExFairness.CrucibleStage

  # Test 1: Behaviour implementation
  describe "behaviour" do
    test "implements Crucible.Stage behaviour" do
      # Verify callbacks exist
      assert function_exported?(CrucibleStage, :run, 2)
      assert function_exported?(CrucibleStage, :describe, 1)
    end
  end

  # Test 2: describe/1 returns map
  describe "describe/1" do
    test "returns map with stage metadata" do
      result = CrucibleStage.describe(%{})
      assert is_map(result)
      assert Map.has_key?(result, :stage)
      assert Map.has_key?(result, :description)
    end
  end

  # Test 3: run/2 with disabled fairness
  describe "run/2 with disabled fairness" do
    test "passes through context when disabled" do
      # ...
    end
  end

  # Test 4: run/2 data extraction
  describe "run/2 data extraction" do
    test "extracts from context.outputs" do
      # ...
    end

    test "extracts from context.assigns" do
      # ...
    end

    test "returns error when no data available" do
      # ...
    end
  end

  # Test 5: run/2 metric computation
  describe "run/2 metric computation" do
    test "computes demographic parity" do
      # ...
    end

    test "computes multiple metrics" do
      # ...
    end

    test "computes calibration with probabilities" do
      # ...
    end
  end

  # Test 6: run/2 result merging
  describe "run/2 result merging" do
    test "stores results in context.metrics.fairness" do
      # ...
    end

    test "includes overall_passes and violations" do
      # ...
    end
  end

  # Test 7: error handling
  describe "run/2 error handling" do
    test "returns error for invalid context structure" do
      # ...
    end

    test "handles metric computation failures gracefully" do
      # ...
    end
  end
end
```

### Implementation Steps

1. **Create test file first** - Write all test cases
2. **Run tests** - Verify they fail (red)
3. **Create module** - Implement `ExFairness.CrucibleStage`
4. **Implement describe/1** - Return metadata map
5. **Implement run/2 skeleton** - Handle disabled case first
6. **Add data extraction** - From outputs and assigns
7. **Add metric computation** - Reuse ExFairness functions
8. **Add result merging** - Update context.metrics
9. **Run tests** - Verify they pass (green)
10. **Refactor** - Clean up, add typespecs

### Quality Requirements

Before considering complete, verify:

```bash
# All tests pass
cd /home/home/p/g/North-Shore-AI/ExFairness
mix test

# No warnings
mix compile --warnings-as-errors

# Dialyzer clean
mix dialyzer

# Credo strict
mix credo --strict

# Check coverage
mix coveralls
```

### Update README.md

Add section about Crucible.Stage integration:

```markdown
### Pipeline Integration

ExFairness provides a `Crucible.Stage` implementation for integration with
crucible_framework pipelines:

\`\`\`elixir
# Add to pipeline
stages = [
  {ExFairness.CrucibleStage, %{metrics: [:demographic_parity, :equalized_odds]}}
]

# Context requirements
context = %Crucible.Context{
  experiment: %{
    reliability: %{
      fairness: %CrucibleIR.Reliability.Fairness{
        enabled: true,
        metrics: [:demographic_parity],
        group_by: :gender,
        threshold: 0.1
      }
    }
  },
  outputs: [...],  # or use assigns
  metrics: %{}
}

{:ok, result} = ExFairness.CrucibleStage.run(context, %{})
# result.metrics.fairness contains evaluation results
\`\`\`
```

## Reference: CrucibleIR.Reliability.Fairness struct

From crucible_ir (used in tests and existing stage):

```elixir
%CrucibleIR.Reliability.Fairness{
  enabled: true,              # Enable/disable evaluation
  metrics: [],                # List of metric atoms
  group_by: :gender,          # Sensitive attribute field name
  threshold: 0.1,             # Pass/fail threshold
  fail_on_violation: false,   # Error on violations?
  options: %{}                # Additional options
}
```

## Reference: Crucible.Context struct

From crucible_framework:

```elixir
%Crucible.Context{
  experiment: experiment,     # Experiment config
  examples: [],               # Input examples
  outputs: [],                # Model outputs
  metrics: %{},               # Accumulated metrics
  assigns: %{},               # Arbitrary assigns
  status: :pending            # Execution status
}
```

## Summary

The implementation should:

1. Create `ExFairness.CrucibleStage` implementing `@behaviour Crucible.Stage`
2. Extract predictions/labels/sensitive_attr from context outputs or assigns
3. Run configured fairness metrics using existing ExFairness functions
4. Merge results into `context.metrics.fairness`
5. Return `{:ok, updated_context}` or `{:error, reason}`
6. Include comprehensive tests following TDD
7. Pass all quality gates (warnings, dialyzer, credo)
8. Update README.md with integration documentation

This creates a proper bridge between ExFairness and crucible_framework pipelines.
