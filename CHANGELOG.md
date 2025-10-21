# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.3.0
- Statistical inference (bootstrap confidence intervals, hypothesis testing)
- Calibration fairness metric
- Intersectional fairness analysis
- Threshold optimization (post-processing mitigation)
- Integration tests with real datasets (Adult, COMPAS, German Credit)
- Property-based testing with StreamData
- Performance benchmarking suite

## [0.2.0] - 2025-10-20

### Added - Comprehensive Technical Documentation
- **future_directions.md (1,941 lines)** - Complete roadmap to v1.0.0
  - Detailed specifications for statistical inference
  - Calibration metric with complete algorithm
  - Intersectional analysis implementation plan
  - Threshold optimization algorithm
  - 6-month development timeline
  - 12+ additional research citations
- **implementation_report.md (1,288 lines)** - Technical implementation details
  - Module-by-module analysis of all 14 modules
  - Algorithm documentation with pseudocode
  - Design decisions and rationale
  - Performance characteristics
  - Code statistics and metrics
- **testing_and_qa_strategy.md (1,220 lines)** - QA methodology
  - TDD philosophy and evidence
  - Complete test coverage matrix (134 tests)
  - Edge case testing strategy
  - Future testing enhancements (property testing, integration testing)
  - Quality gates and CI/CD specifications

### Enhanced - README.md
- Expanded from ~660 to 1,437 lines (+118%)
- Added **Mathematical Foundations** section (200+ lines)
  - Complete mathematical definitions for all 4 metrics
  - Formal probability notation
  - Disparity measures
  - Comprehensive citations with DOI numbers
- Added **Theoretical Background** section (300+ lines)
  - Types of fairness (group, individual, causal)
  - Measurement problem discussion
  - Impossibility theorem with proof intuition
  - Fairness-accuracy tradeoff analysis
- Added **Advanced Usage** section (200+ lines)
  - Axon integration example (neural networks)
  - Scholar integration example (classical ML)
  - Batch fairness analysis
  - Production monitoring with GenServer
- Expanded **Research Foundations** (150+ lines)
  - 15+ peer-reviewed papers with full bibliographic details
  - DOI numbers for all citations
  - Framework comparisons (AIF360, Fairlearn, etc.)
- Added **API Reference** section
- Updated real-world use cases with legal compliance checks

### Documentation
- Total documentation: ~9,120 lines
- Academic citations: 27+ peer-reviewed papers
- Working code examples: 20+
- Integration patterns documented

## [0.1.0] - 2025-10-20

### Added - Core Implementation

**Infrastructure:**
- `ExFairness.Error` - Custom exception handling with type safety
- `ExFairness.Validation` - Comprehensive input validation
  - Binary tensor validation
  - Shape matching validation
  - Multiple groups requirement (min 2 groups)
  - Sufficient samples validation (default: 10 per group)
  - Helpful error messages with actionable suggestions
- `ExFairness.Utils` - GPU-accelerated tensor operations
  - `positive_rate/2` - Positive prediction rate with masking
  - `create_group_mask/2` - Binary mask generation
  - `group_count/2` - Sample counting per group
  - `group_positive_rates/2` - Batch rate computation
- `ExFairness.Utils.Metrics` - Classification metrics
  - `confusion_matrix/3` - TP, FP, TN, FN with masking
  - `true_positive_rate/3` - TPR/Recall
  - `false_positive_rate/3` - FPR
  - `positive_predictive_value/3` - PPV/Precision

**Fairness Metrics:**
- `ExFairness.Metrics.DemographicParity` - P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
  - Configurable threshold (default: 0.1)
  - Plain language interpretations
  - Citations: Dwork et al. (2012), Feldman et al. (2015)
- `ExFairness.Metrics.EqualizedOdds` - Equal TPR and FPR across groups
  - Both error rates checked
  - Combined pass/fail determination
  - Citations: Hardt et al. (2016)
- `ExFairness.Metrics.EqualOpportunity` - Equal TPR across groups
  - Relaxed version of equalized odds
  - Focus on false negative parity
  - Citations: Hardt et al. (2016)
- `ExFairness.Metrics.PredictiveParity` - Equal PPV across groups
  - Precision parity
  - Consistent prediction meaning
  - Citations: Chouldechova (2017)

**Detection Algorithms:**
- `ExFairness.Detection.DisparateImpact` - EEOC 80% rule
  - Legal standard for adverse impact
  - 4/5ths rule implementation
  - Legal interpretation with EEOC context
  - Citations: EEOC (1978), Biddle (2006)

**Mitigation Techniques:**
- `ExFairness.Mitigation.Reweighting` - Sample weighting for fairness
  - Supports demographic parity and equalized odds targets
  - Formula: w(a,y) = P(Y=y) / P(A=a,Y=y)
  - Normalized weights (mean = 1.0)
  - GPU-accelerated via Nx.Defn
  - Citations: Kamiran & Calders (2012)

**Reporting System:**
- `ExFairness.Report` - Multi-metric fairness assessment
  - Aggregate pass/fail counts
  - Overall assessment generation
  - Markdown export (human-readable)
  - JSON export (machine-readable)

**Main API:**
- `ExFairness.demographic_parity/3` - Convenience function
- `ExFairness.equalized_odds/4` - Convenience function
- `ExFairness.equal_opportunity/4` - Convenience function
- `ExFairness.predictive_parity/4` - Convenience function
- `ExFairness.fairness_report/4` - Comprehensive reporting

### Testing
- 134 total tests (102 unit tests + 32 doctests)
- 100% pass rate
- Comprehensive edge case coverage
- Strict TDD approach (Red-Green-Refactor)
- All tests async (parallel execution)

### Quality Gates
- Zero compiler warnings (enforced)
- Zero Dialyzer errors (type-safe)
- Credo strict mode configured
- Code formatting enforced (100 char lines)
- ExCoveralls configured for coverage reports

### Documentation
- Comprehensive README.md with examples
- Complete module documentation (@moduledoc)
- Complete function documentation (@doc)
- Working examples (verified by doctests)
- Research citations in all metrics
- Mathematical definitions included

### Dependencies
- Production: `nx ~> 0.7` (only production dependency)
- Development: `ex_doc`, `dialyxir`, `excoveralls`, `credo`, `stream_data`, `jason`

---

[Unreleased]: https://github.com/North-Shore-AI/ExFairness/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/North-Shore-AI/ExFairness/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/North-Shore-AI/ExFairness/releases/tag/v0.1.0
