defmodule ExFairness.Utils.Bootstrap do
  @moduledoc """
  Bootstrap confidence interval computation for fairness metrics.

  Implements stratified bootstrap to preserve group proportions and
  parallel computation for performance.

  ## Algorithm

  Bootstrap resampling provides non-parametric confidence intervals without
  distributional assumptions:

  1. Compute observed metric: M_obs = M(data)
  2. For i = 1 to B (bootstrap samples):
     a. Sample n datapoints with replacement: data*_i
     b. Compute M*_i = M(data*_i)
  3. Sort {M*_1, ..., M*_B}
  4. CI_lower = percentile(α/2)
     CI_upper = percentile(1 - α/2)

  ## Stratified Bootstrap

  To preserve group proportions, sample separately from each group:
  - Sample n_A from group A with replacement
  - Sample n_B from group B with replacement
  - Combine samples and compute metric

  ## References

  - Efron, B., & Tibshirani, R. J. (1994). "An introduction to the
    bootstrap." CRC press.
  - Davison, A. C., & Hinkley, D. V. (1997). "Bootstrap methods and
    their application." Cambridge university press.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> metric_fn = fn [preds, sens] ->
      ...>   result = ExFairness.demographic_parity(preds, sens)
      ...>   result.disparity
      ...> end
      iex> result = ExFairness.Utils.Bootstrap.confidence_interval(
      ...>   [predictions, sensitive],
      ...>   metric_fn,
      ...>   n_samples: 100
      ...> )
      iex> {lower, upper} = result.confidence_interval
      iex> is_float(lower) and is_float(upper) and lower <= upper
      true

  """

  @default_n_samples 1000
  @default_confidence_level 0.95
  @default_method :percentile

  @type bootstrap_result :: %{
          point_estimate: float(),
          confidence_interval: {float(), float()},
          confidence_level: float(),
          n_samples: integer(),
          method: :percentile | :basic
        }

  @doc """
  Computes bootstrap confidence interval for a fairness metric.

  ## Parameters

    * `data` - List of tensors [predictions, labels?, sensitive_attr]
    * `metric_fn` - Function computing the metric on data
    * `opts` - Options:
      * `:n_samples` - Number of bootstrap samples (default: 1000)
      * `:confidence_level` - Confidence level (default: 0.95)
      * `:method` - Bootstrap method (:percentile or :basic, default: :percentile)
      * `:stratified` - Preserve group proportions (default: true)
      * `:parallel` - Use parallel computation (default: true)
      * `:seed` - Random seed for reproducibility (default: system time)

  ## Returns

  Map containing point estimate and confidence interval:
    * `:point_estimate` - Observed metric value
    * `:confidence_interval` - Tuple {lower, upper}
    * `:confidence_level` - Confidence level used
    * `:n_samples` - Number of bootstrap samples
    * `:method` - Bootstrap method used

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> metric_fn = fn [preds, sens] ->
      ...>   result = ExFairness.demographic_parity(preds, sens)
      ...>   result.disparity
      ...> end
      iex> result = ExFairness.Utils.Bootstrap.confidence_interval(
      ...>   [predictions, sensitive],
      ...>   metric_fn,
      ...>   n_samples: 100, seed: 42
      ...> )
      iex> result.method
      :percentile

  """
  @spec confidence_interval([Nx.Tensor.t()], function(), keyword()) :: bootstrap_result()
  def confidence_interval(data, metric_fn, opts \\ []) do
    n_samples = Keyword.get(opts, :n_samples, @default_n_samples)
    confidence_level = Keyword.get(opts, :confidence_level, @default_confidence_level)
    method = Keyword.get(opts, :method, @default_method)
    stratified = Keyword.get(opts, :stratified, true)
    parallel = Keyword.get(opts, :parallel, true)
    seed = Keyword.get(opts, :seed, :erlang.system_time())

    # Compute observed statistic
    point_estimate = metric_fn.(data)

    # Get sample size from first tensor
    n = elem(Nx.shape(hd(data)), 0)

    # Generate bootstrap samples and compute statistics
    bootstrap_statistics =
      if parallel do
        compute_parallel_bootstrap(data, metric_fn, n, n_samples, seed, stratified)
      else
        compute_sequential_bootstrap(data, metric_fn, n, n_samples, seed, stratified)
      end
      |> Enum.sort()

    # Compute confidence interval based on method
    ci =
      case method do
        :percentile -> percentile_ci(bootstrap_statistics, confidence_level)
        :basic -> basic_ci(bootstrap_statistics, point_estimate, confidence_level)
      end

    %{
      point_estimate: point_estimate,
      confidence_interval: ci,
      confidence_level: confidence_level,
      n_samples: n_samples,
      method: method
    }
  end

  # Compute bootstrap statistics in parallel
  @spec compute_parallel_bootstrap(
          [Nx.Tensor.t()],
          function(),
          non_neg_integer(),
          non_neg_integer(),
          integer(),
          boolean()
        ) :: [float()]
  defp compute_parallel_bootstrap(data, metric_fn, n, n_samples, seed, stratified) do
    1..n_samples
    |> Task.async_stream(
      fn i ->
        sample_data = bootstrap_sample(data, n, seed + i, stratified)
        metric_fn.(sample_data)
      end,
      max_concurrency: System.schedulers_online(),
      timeout: :infinity
    )
    |> Enum.map(fn {:ok, stat} -> stat end)
  end

  # Compute bootstrap statistics sequentially
  @spec compute_sequential_bootstrap(
          [Nx.Tensor.t()],
          function(),
          non_neg_integer(),
          non_neg_integer(),
          integer(),
          boolean()
        ) :: [float()]
  defp compute_sequential_bootstrap(data, metric_fn, n, n_samples, seed, stratified) do
    for i <- 1..n_samples do
      sample_data = bootstrap_sample(data, n, seed + i, stratified)
      metric_fn.(sample_data)
    end
  end

  # Generate a single bootstrap sample
  @spec bootstrap_sample([Nx.Tensor.t()], non_neg_integer(), integer(), boolean()) ::
          [Nx.Tensor.t()]
  defp bootstrap_sample(data, n, seed, stratified) do
    if stratified and length(data) >= 2 do
      # Stratified sampling: preserve group proportions
      # Last tensor is assumed to be sensitive attribute
      sensitive_attr = List.last(data)
      stratified_sample(data, sensitive_attr, n, seed)
    else
      # Simple random sampling with replacement
      simple_sample(data, n, seed)
    end
  end

  # Stratified bootstrap sampling
  @spec stratified_sample([Nx.Tensor.t()], Nx.Tensor.t(), non_neg_integer(), integer()) ::
          [Nx.Tensor.t()]
  defp stratified_sample(data, sensitive_attr, _n, seed) do
    # Get indices for each group
    sensitive_list = Nx.to_flat_list(sensitive_attr)

    group_a_indices =
      sensitive_list
      |> Enum.with_index()
      |> Enum.filter(fn {val, _idx} -> val == 0 end)
      |> Enum.map(fn {_val, idx} -> idx end)

    group_b_indices =
      sensitive_list
      |> Enum.with_index()
      |> Enum.filter(fn {val, _idx} -> val == 1 end)
      |> Enum.map(fn {_val, idx} -> idx end)

    n_a = length(group_a_indices)
    n_b = length(group_b_indices)

    # Sample from each group with replacement
    :rand.seed(:exsss, seed)

    sampled_a_indices =
      for _ <- 1..n_a do
        Enum.random(group_a_indices)
      end

    sampled_b_indices =
      for _ <- 1..n_b do
        Enum.random(group_b_indices)
      end

    # Combine samples
    all_sampled_indices = sampled_a_indices ++ sampled_b_indices

    # Gather sampled data
    Enum.map(data, fn tensor ->
      sampled_values =
        Enum.map(all_sampled_indices, fn idx ->
          Nx.to_flat_list(tensor) |> Enum.at(idx)
        end)

      Nx.tensor(sampled_values)
    end)
  end

  # Simple bootstrap sampling with replacement
  @spec simple_sample([Nx.Tensor.t()], non_neg_integer(), integer()) :: [Nx.Tensor.t()]
  defp simple_sample(data, n, seed) do
    :rand.seed(:exsss, seed)
    indices = for _ <- 1..n, do: :rand.uniform(n) - 1

    Enum.map(data, fn tensor ->
      flat_list = Nx.to_flat_list(tensor)

      sampled_values =
        Enum.map(indices, fn idx ->
          Enum.at(flat_list, idx)
        end)

      Nx.tensor(sampled_values)
    end)
  end

  # Compute percentile bootstrap confidence interval
  @spec percentile_ci([float()], float()) :: {float(), float()}
  defp percentile_ci(bootstrap_values, confidence_level) do
    n = length(bootstrap_values)
    alpha = 1 - confidence_level

    lower_idx = floor(n * alpha / 2) |> max(0)
    upper_idx = (ceil(n * (1 - alpha / 2)) - 1) |> min(n - 1)

    lower = Enum.at(bootstrap_values, lower_idx)
    upper = Enum.at(bootstrap_values, upper_idx)

    {lower, upper}
  end

  # Compute basic bootstrap confidence interval
  @spec basic_ci([float()], float(), float()) :: {float(), float()}
  defp basic_ci(bootstrap_values, observed, confidence_level) do
    {boot_lower, boot_upper} = percentile_ci(bootstrap_values, confidence_level)

    # Basic bootstrap: 2*observed - percentiles (reversed)
    lower = 2 * observed - boot_upper
    upper = 2 * observed - boot_lower

    {lower, upper}
  end
end
