# Contributing to ExFairness

Thank you for your interest in contributing to ExFairness! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Guidelines](#contribution-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Expected Behavior

- Be respectful and considerate in all interactions
- Provide constructive feedback
- Focus on what's best for the project and community
- Show empathy towards other contributors

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks or trolling
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## Getting Started

### Prerequisites

- Elixir 1.14 or higher
- Erlang/OTP 25 or higher
- Git
- Basic understanding of fairness in machine learning (optional but helpful)

### Setting Up Development Environment

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ExFairness.git
cd ExFairness

# 3. Add upstream remote
git remote add upstream https://github.com/North-Shore-AI/ExFairness.git

# 4. Install dependencies
mix deps.get

# 5. Verify tests pass
mix test

# 6. Verify quality checks pass
mix format --check-formatted
mix compile --warnings-as-errors
mix credo --strict
```

---

## Development Workflow

### Strict Test-Driven Development (TDD)

ExFairness follows **strict TDD**. All contributions must follow the Red-Green-Refactor cycle:

#### 1. RED Phase - Write Failing Tests

```elixir
# test/ex_fairness/metrics/new_metric_test.exs
defmodule ExFairness.Metrics.NewMetricTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Metrics.NewMetric

  describe "compute/3" do
    test "computes metric correctly" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      result = NewMetric.compute(predictions, sensitive)

      assert result.metric_value == expected_value
      assert result.passes == expected_pass_fail
    end

    # Add more tests...
  end
end
```

Run tests to verify they fail:
```bash
mix test test/ex_fairness/metrics/new_metric_test.exs
# Should show compilation error or test failures
```

#### 2. GREEN Phase - Implement to Pass

```elixir
# lib/ex_fairness/metrics/new_metric.ex
defmodule ExFairness.Metrics.NewMetric do
  @moduledoc """
  Documentation for new metric.

  ## Mathematical Definition

  [Include formal definition]

  ## When to Use

  [Explain appropriate use cases]

  ## Limitations

  [Discuss limitations]

  ## References

  [Include research citations]
  """

  alias ExFairness.Validation

  @spec compute(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: map()
  def compute(predictions, sensitive_attr, opts \\ []) do
    # Validate inputs
    Validation.validate_predictions!(predictions)
    # ... implement logic
  end
end
```

Run tests to verify they pass:
```bash
mix test test/ex_fairness/metrics/new_metric_test.exs
# Should show all tests passing
```

#### 3. REFACTOR Phase - Optimize and Document

- Add comprehensive documentation
- Add type specifications
- Optimize performance
- Add doctests
- Ensure code formatting

```bash
mix format
mix compile --warnings-as-errors
mix credo --strict
```

---

## Contribution Guidelines

### Types of Contributions

We welcome:

1. **Bug Fixes** - Fix issues in existing code
2. **New Metrics** - Implement additional fairness metrics
3. **New Detection Algorithms** - Add bias detection methods
4. **New Mitigation Techniques** - Add fairness mitigation approaches
5. **Documentation Improvements** - Enhance docs, examples, guides
6. **Performance Optimizations** - Improve speed/memory usage
7. **Test Additions** - Add edge cases, property tests, integration tests

### Before Starting

1. **Check existing issues** - Avoid duplicate work
2. **Open an issue** - Discuss your proposal first
3. **Get approval** - Especially for large changes
4. **Follow the roadmap** - See `docs/20251020/future_directions.md`

### Coding Standards

#### Code Style

- Follow the [Elixir Style Guide](https://github.com/christopheradams/elixir_style_guide)
- Use `mix format` (configured for 100-char lines)
- Pass `mix credo --strict`
- No compiler warnings

#### Naming Conventions

```elixir
# Modules: CamelCase
defmodule ExFairness.Metrics.DemographicParity

# Functions: snake_case
def compute_disparity(predictions, sensitive_attr)

# Variables: snake_case
group_a_rate = 0.5

# Constants: @uppercase
@default_threshold 0.1

# Private functions: prefix with defp
defp generate_interpretation(...)
```

#### Type Specifications

**Required for all public functions:**

```elixir
@type result :: %{
  disparity: float(),
  passes: boolean(),
  threshold: float()
}

@spec compute(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: result()
def compute(predictions, sensitive_attr, opts \\ []) do
  # ...
end
```

---

## Testing Requirements

### Minimum Test Coverage

Every new feature must include:

1. **At least 5 unit tests:**
   - Happy path (normal case)
   - Edge case #1
   - Edge case #2
   - Error case (validation)
   - Configuration test (custom options)

2. **At least 1 doctest:**
   - Working example in @doc
   - Verified to execute correctly

3. **Property tests (if applicable):**
   - For metrics: symmetry, boundedness, monotonicity

### Test Data Requirements

- **Minimum 10 samples per group** (statistical reliability)
- **Use 20-element patterns** for consistency
- **Explicit calculations** in comments
- **Realistic scenarios** (not trivial 1-2 samples)

Example:
```elixir
test "computes metric correctly" do
  # Group A: 5/10 = 0.5, Group B: 3/10 = 0.3
  # Expected disparity: 0.2
  predictions = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
  sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

  result = YourMetric.compute(predictions, sensitive)

  assert_in_delta(result.disparity, 0.2, 0.01)
end
```

### Running Tests

```bash
# Run all tests
mix test

# Run specific test file
mix test test/ex_fairness/metrics/your_metric_test.exs

# Run with coverage
mix coveralls

# Run specific test
mix test test/ex_fairness/metrics/your_metric_test.exs:42
```

---

## Documentation Standards

### Module Documentation (@moduledoc)

Every module must include:

```elixir
defmodule ExFairness.Metrics.YourMetric do
  @moduledoc """
  Brief description of the metric.

  ## Mathematical Definition

  [Include formal probability notation]

  ## When to Use

  - Use case 1
  - Use case 2

  ## Limitations

  - Limitation 1
  - Limitation 2

  ## References

  - Author (Year). "Paper title." *Venue*.

  ## Examples

      iex> # Working example
      iex> result = ExFairness.Metrics.YourMetric.compute(...)
      iex> result.passes
      true

  """
end
```

### Function Documentation (@doc)

Every public function must include:

```elixir
@doc """
Brief description.

## Parameters

  * `param1` - Description
  * `param2` - Description
  * `opts` - Options:
    * `:option1` - Description (default: value)

## Returns

A map containing:
  * `:field1` - Description
  * `:field2` - Description

## Examples

    iex> result = function(arg1, arg2)
    iex> result.field1
    expected_value

"""
@spec function(type1(), type2(), keyword()) :: return_type()
def function(param1, param2, opts \\ []) do
  # Implementation
end
```

### Citation Format

Follow academic citation standards:

```
Author, A., Author, B., & Author, C. (Year). "Title of paper."
*Journal/Conference Name*, volume(issue), pages.
DOI: xx.xxxx/xxxxx
```

Example:
```
Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity
in Supervised Learning." In *Advances in Neural Information Processing
Systems* (NeurIPS '16), pp. 3315-3323.
```

---

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow TDD (tests first)
   - Follow coding standards
   - Update documentation

3. **Verify quality**
   ```bash
   mix format
   mix test
   mix compile --warnings-as-errors
   mix credo --strict
   mix dialyzer  # If PLT already built
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "Add calibration fairness metric

   Implements calibration metric as specified in Pleiss et al. (2017).
   Includes binning, ECE computation, and calibration curves.

   - 15 unit tests
   - 2 doctests
   - Complete documentation with mathematical definition
   - Citations included
   "
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open Pull Request**
   - Use clear PR title
   - Reference any related issues
   - Describe what you changed and why
   - Include test results

### Pull Request Template

```markdown
## Description
[Describe your changes]

## Motivation
[Why is this change needed?]

## Related Issues
Fixes #123

## Changes
- [ ] New feature / bug fix / documentation
- [ ] Tests added/updated
- [ ] Documentation added/updated
- [ ] CHANGELOG.md updated

## Testing
- [ ] All tests pass (`mix test`)
- [ ] No warnings (`mix compile --warnings-as-errors`)
- [ ] Credo passes (`mix credo --strict`)
- [ ] Code formatted (`mix format --check-formatted`)

## Checklist
- [ ] Followed TDD (tests written first)
- [ ] Added type specs (@spec)
- [ ] Added documentation (@doc)
- [ ] Included research citations (if applicable)
- [ ] Updated CHANGELOG.md
```

### Commit Message Guidelines

**Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

**Example:**
```
feat: Add calibration fairness metric

Implements calibration metric with binning and ECE computation.
Based on Pleiss et al. (2017) "On fairness and calibration."

- 15 unit tests for binning strategies and edge cases
- 2 doctests with working examples
- Complete mathematical documentation
- Citations: Pleiss et al. (2017)

Closes #42
```

---

## Adding New Fairness Metrics

### Step-by-Step Guide

#### 1. Research Phase

- [ ] Find peer-reviewed paper defining the metric
- [ ] Understand mathematical definition
- [ ] Identify when to use and limitations
- [ ] Check if similar metric exists

#### 2. Design Phase

- [ ] Write specification document (in `docs/`)
- [ ] Define function signature and return type
- [ ] Plan test cases (minimum 10)
- [ ] Get approval via GitHub issue

#### 3. Implementation Phase (TDD)

**RED - Write tests first:**

```bash
# Create test file
touch test/ex_fairness/metrics/your_metric_test.exs

# Write comprehensive tests
# Run and verify they fail
mix test test/ex_fairness/metrics/your_metric_test.exs
```

**GREEN - Implement:**

```bash
# Create implementation file
touch lib/ex_fairness/metrics/your_metric.ex

# Implement minimum code to pass tests
# Run and verify tests pass
mix test test/ex_fairness/metrics/your_metric_test.exs
```

**REFACTOR - Polish:**

```bash
# Add documentation
# Add type specs
# Optimize if needed
# Add to main API (lib/ex_fairness.ex)

# Verify everything passes
mix test
mix format
mix compile --warnings-as-errors
mix credo --strict
```

#### 4. Documentation Phase

- [ ] Add to README.md examples section
- [ ] Add to mathematical foundations section
- [ ] Include in metrics reference table
- [ ] Add research citations with DOI
- [ ] Update CHANGELOG.md

#### 5. Validation Phase

- [ ] Test against reference implementation (if available)
- [ ] Verify on real dataset (if applicable)
- [ ] Performance benchmark
- [ ] Code review

### Metric Template

Use this template for new metrics:

```elixir
defmodule ExFairness.Metrics.YourMetric do
  @moduledoc """
  Brief description.

  ## Mathematical Definition

  [Formal definition with notation]

  ## When to Use

  - Use case 1
  - Use case 2

  ## Limitations

  - Limitation 1
  - Limitation 2

  ## References

  - Citation 1
  - Citation 2

  ## Examples

      iex> # Working example
  """

  alias ExFairness.{Utils, Validation}

  @default_threshold 0.1
  @default_min_per_group 10

  @type result :: %{
    # Define return type fields
  }

  @spec compute(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: result()
  def compute(predictions, sensitive_attr, opts \\ []) do
    # 1. Extract options
    # 2. Validate inputs
    # 3. Compute metric
    # 4. Generate interpretation
    # 5. Return result map
  end

  defp generate_interpretation(...) do
    # Plain language explanation
  end
end
```

---

## Testing Requirements

### Test File Structure

```elixir
defmodule ExFairness.Metrics.YourMetricTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Metrics.YourMetric

  alias ExFairness.Metrics.YourMetric

  describe "compute/3" do
    test "computes perfect fairness" do
      # Test with zero disparity
    end

    test "detects disparity" do
      # Test with known disparity
    end

    test "accepts custom threshold" do
      # Test configuration options
    end

    test "validates inputs" do
      # Test input validation
    end

    test "handles edge case: all zeros" do
      # Edge case testing
    end

    test "handles edge case: all ones" do
      # Edge case testing
    end

    test "returns interpretation" do
      # Test interpretation generation
    end
  end
end
```

### Mandatory Test Coverage

- [ ] Happy path (normal operation)
- [ ] Perfect fairness (disparity = 0)
- [ ] Maximum disparity
- [ ] Custom threshold
- [ ] Input validation (invalid inputs raise errors)
- [ ] Edge case: all zeros
- [ ] Edge case: all ones
- [ ] Edge case: single value
- [ ] Unbalanced groups
- [ ] Interpretation generation

### Assertion Guidelines

**For floating point values:**
```elixir
# Use assert_in_delta with 0.01 tolerance
assert_in_delta(result.disparity, 0.5, 0.01)
```

**For exact values:**
```elixir
# Use exact equality
assert result.passes == true
assert Nx.to_number(count) == 10
```

**For errors:**
```elixir
# Use assert_raise with regex
assert_raise ExFairness.Error, ~r/must be binary/, fn ->
  YourMetric.compute(invalid_input, sensitive)
end
```

---

## Documentation Standards

### Required Documentation Elements

Every new module must include:

1. **@moduledoc with:**
   - Brief description
   - Mathematical definition (formal notation)
   - When to use (3+ bullet points)
   - Limitations (2+ bullet points)
   - Research citations (full bibliographic info)
   - Working example (doctest)

2. **@doc for every public function with:**
   - Description
   - Parameters section (with types and defaults)
   - Returns section (with structure)
   - Examples section (with doctest)

3. **@spec for every public function**

4. **Inline comments for complex logic**

### Documentation Verification

```bash
# Generate docs locally
mix docs

# Open in browser
open doc/index.html

# Check for warnings
mix docs 2>&1 | grep warning

# Verify doctests pass
mix test --only doctest
```

---

## Code Review Checklist

Before submitting PR, verify:

### Code Quality
- [ ] No compiler warnings (`mix compile --warnings-as-errors`)
- [ ] No Credo issues (`mix credo --strict`)
- [ ] Code formatted (`mix format --check-formatted`)
- [ ] No Dialyzer errors (`mix dialyzer`)

### Testing
- [ ] All new code has tests
- [ ] All tests pass (`mix test`)
- [ ] Test coverage is comprehensive
- [ ] Edge cases covered
- [ ] Doctests work

### Documentation
- [ ] @moduledoc added to new modules
- [ ] @doc added to new public functions
- [ ] @spec added to all public functions
- [ ] Examples work (verified by doctests)
- [ ] Research citations included
- [ ] README.md updated (if user-facing change)
- [ ] CHANGELOG.md updated

### Quality
- [ ] Follows existing code patterns
- [ ] No code duplication
- [ ] Appropriate use of Nx.Defn (GPU acceleration)
- [ ] Error messages are helpful
- [ ] Comments explain "why" not "what"

---

## Development Commands

### Essential Commands

```bash
# Install dependencies
mix deps.get

# Run tests
mix test

# Run specific test
mix test test/path/to/test.exs:line_number

# Run with coverage
mix coveralls
mix coveralls.html  # HTML report in cover/

# Format code
mix format

# Check formatting
mix format --check-formatted

# Compile with warnings as errors
mix compile --warnings-as-errors

# Run linter
mix credo --strict

# Type checking (requires PLT build)
mix dialyzer

# Generate documentation
mix docs

# Full quality check (run before PR)
mix format --check-formatted && \
mix compile --warnings-as-errors && \
mix test && \
mix credo --strict
```

### Building PLT for Dialyzer (One-time)

```bash
# This takes a few minutes the first time
mix dialyzer --plt

# Then run analysis
mix dialyzer
```

---

## Performance Considerations

### When to Use Nx.Defn

**Use for:**
- Numerical computations
- Operations on tensors
- Code that benefits from GPU acceleration

**Don't use for:**
- String manipulation
- Control flow with dynamic decisions
- I/O operations

### Example

```elixir
# Good: Numerical computation with defn
import Nx.Defn

defn compute_disparity(rate_a, rate_b) do
  Nx.abs(Nx.subtract(rate_a, rate_b))
end

# Good: Validation in regular Elixir
def compute(predictions, sensitive_attr, opts \\ []) do
  Validation.validate_predictions!(predictions)  # Regular Elixir
  disparity = compute_disparity(rate_a, rate_b)  # Nx.Defn
end
```

---

## Adding Research Citations

### Citation Requirements

For new metrics or algorithms:

1. **Find the original paper** that proposed the technique
2. **Include full citation** with:
   - Authors (all, or first 3 + "et al.")
   - Year
   - Title (in quotes)
   - Venue (journal or conference)
   - Volume/issue/pages (for journals)
   - DOI (if available)

3. **Add to module @moduledoc**
4. **Add to README.md** Research Foundations section

### Citation Format Example

```elixir
@moduledoc """
Your metric description.

## References

- Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity
  in Supervised Learning." In *Advances in Neural Information Processing
  Systems* (NeurIPS '16), pp. 3315-3323.
"""
```

---

## Common Pitfalls to Avoid

### Don't

❌ Write implementation before tests
❌ Change tests to make them pass (fix code instead)
❌ Skip edge case testing
❌ Use floating point equality (use `assert_in_delta`)
❌ Forget to update CHANGELOG.md
❌ Add compiler warnings
❌ Skip documentation
❌ Use trivial test data (2-3 samples)
❌ Forget type specifications
❌ Copy-paste without attribution

### Do

✅ Write tests first (TDD)
✅ Use `assert_in_delta` for floats
✅ Test edge cases explicitly
✅ Update CHANGELOG.md
✅ Add comprehensive documentation
✅ Include research citations
✅ Use realistic test data (10+ per group)
✅ Add type specifications
✅ Format code before committing
✅ Run full quality check before PR

---

## Getting Help

### Resources

- **Documentation:** https://hexdocs.pm/ex_fairness
- **Issues:** https://github.com/North-Shore-AI/ExFairness/issues
- **Discussions:** https://github.com/North-Shore-AI/ExFairness/discussions
- **Technical Docs:** `docs/20251020/` directory

### Asking Questions

**Good question:**
> "I want to add the calibration metric from Pleiss et al. (2017). I've read the paper and understand the math. Should I use uniform binning or quantile binning for the default? The paper uses uniform but some implementations use quantile."

**Contains:**
- Specific feature
- Research reference
- Shows you've done homework
- Asks specific question

**Not helpful:**
> "How do I add a new metric?"

**Too vague:**
- No specific metric mentioned
- No research reference
- No specific question

### Response Time

- Simple questions: 24-48 hours
- Feature proposals: 3-7 days for review
- Pull requests: 1-2 weeks for review

---

## Release Process (Maintainers Only)

### Version Numbering

Follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes only

### Release Checklist

- [ ] All tests pass
- [ ] CHANGELOG.md updated
- [ ] Version bumped in mix.exs
- [ ] Documentation generated successfully
- [ ] Git tag created (`git tag -a v0.2.0 -m "Release v0.2.0"`)
- [ ] Pushed to GitHub (`git push --tags`)
- [ ] Published to Hex.pm (`mix hex.publish`)
- [ ] HexDocs generated
- [ ] GitHub release created with notes

---

## Recognition

Contributors will be:

- Listed in release notes
- Mentioned in CHANGELOG.md
- Credited in git commit history
- Thanked in project documentation

Significant contributions may lead to:

- Co-authorship on academic papers
- Maintainer status
- Conference presentation opportunities

---

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing issues
3. Open a new issue with the `question` label
4. Be patient - we're a small team!

---

## Thank You!

Your contributions help make ML fairer for everyone. We appreciate your effort to improve ExFairness!

**Happy Contributing!** 🚀

---

**Last Updated:** October 20, 2025
**Version:** 1.0
**Maintainers:** North Shore AI Research Team
