defmodule ExFairness.Validation do
  @moduledoc """
  Input validation utilities for ExFairness.

  Provides comprehensive validation for tensors used in fairness metrics,
  ensuring data quality and providing helpful error messages.
  """

  @doc """
  Validates predictions tensor.

  ## Parameters

    * `predictions` - Tensor to validate

  ## Returns

  The validated predictions tensor.

  ## Raises

  `ExFairness.Error` if validation fails.

  ## Examples

      iex> predictions = Nx.tensor([0, 1, 1, 0])
      iex> result = ExFairness.Validation.validate_predictions!(predictions)
      iex> Nx.to_flat_list(result)
      [0, 1, 1, 0]

  """
  @spec validate_predictions!(Nx.Tensor.t()) :: Nx.Tensor.t()
  def validate_predictions!(predictions) do
    validate_tensor!(predictions, "predictions")
    validate_binary!(predictions, "predictions")
    validate_non_empty!(predictions, "predictions")

    predictions
  end

  @doc """
  Validates labels tensor.

  ## Parameters

    * `labels` - Tensor to validate

  ## Returns

  The validated labels tensor.

  ## Raises

  `ExFairness.Error` if validation fails.

  ## Examples

      iex> labels = Nx.tensor([0, 1, 1, 0])
      iex> result = ExFairness.Validation.validate_labels!(labels)
      iex> Nx.to_flat_list(result)
      [0, 1, 1, 0]

  """
  @spec validate_labels!(Nx.Tensor.t()) :: Nx.Tensor.t()
  def validate_labels!(labels) do
    validate_tensor!(labels, "labels")
    validate_binary!(labels, "labels")
    validate_non_empty!(labels, "labels")

    labels
  end

  @doc """
  Validates sensitive attribute tensor.

  ## Parameters

    * `sensitive_attr` - Tensor to validate
    * `opts` - Options:
      * `:min_per_group` - Minimum samples per group (default: 10)

  ## Returns

  The validated sensitive attribute tensor.

  ## Raises

  `ExFairness.Error` if validation fails.

  ## Examples

      iex> sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Validation.validate_sensitive_attr!(sensitive_attr)
      iex> Nx.size(result)
      20

  """
  @spec validate_sensitive_attr!(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def validate_sensitive_attr!(sensitive_attr, opts \\ []) do
    min_per_group = Keyword.get(opts, :min_per_group, 10)

    validate_tensor!(sensitive_attr, "sensitive_attr")
    validate_binary!(sensitive_attr, "sensitive_attr")
    validate_non_empty!(sensitive_attr, "sensitive_attr")
    validate_multiple_groups!(sensitive_attr)
    validate_sufficient_samples!(sensitive_attr, min_per_group)

    sensitive_attr
  end

  @doc """
  Validates that tensors have matching shapes.

  ## Parameters

    * `tensors` - List of tensors to validate
    * `names` - List of tensor names for error messages

  ## Returns

  The list of validated tensors.

  ## Raises

  `ExFairness.Error` if shapes don't match.

  ## Examples

      iex> t1 = Nx.tensor([1, 2, 3])
      iex> t2 = Nx.tensor([4, 5, 6])
      iex> [r1, r2] = ExFairness.Validation.validate_matching_shapes!([t1, t2], ["t1", "t2"])
      iex> {Nx.to_flat_list(r1), Nx.to_flat_list(r2)}
      {[1, 2, 3], [4, 5, 6]}

  """
  @spec validate_matching_shapes!([Nx.Tensor.t()], [String.t()]) :: [Nx.Tensor.t()]
  def validate_matching_shapes!(tensors, names) do
    shapes = Enum.map(tensors, &Nx.shape/1)

    unless Enum.all?(shapes, fn s -> s == hd(shapes) end) do
      shape_info =
        Enum.zip(names, shapes)
        |> Enum.map(fn {n, s} -> "  #{n}: #{inspect(s)}" end)
        |> Enum.join("\n")

      raise ExFairness.Error, """
      Tensor shape mismatch.

      Expected all tensors to have the same shape, but got:
      #{shape_info}
      """
    end

    tensors
  end

  # Private validation functions

  @spec validate_tensor!(any(), String.t()) :: :ok
  defp validate_tensor!(value, name) do
    unless match?(%Nx.Tensor{}, value) do
      raise ExFairness.Error,
            "#{name} must be an Nx.Tensor, got: #{inspect(value)}"
    end

    :ok
  end

  @spec validate_binary!(Nx.Tensor.t(), String.t()) :: :ok
  defp validate_binary!(tensor, name) do
    min_val = Nx.reduce_min(tensor) |> Nx.to_number()
    max_val = Nx.reduce_max(tensor) |> Nx.to_number()

    unless min_val >= 0 and max_val <= 1 do
      raise ExFairness.Error, """
      #{name} must be binary (containing only 0 and 1).

      Found values in range [#{min_val}, #{max_val}].
      """
    end

    :ok
  end

  @spec validate_non_empty!(Nx.Tensor.t(), String.t()) :: :ok
  defp validate_non_empty!(tensor, name) do
    size = Nx.size(tensor)

    if size == 0 do
      raise ExFairness.Error, "#{name} cannot be empty"
    end

    :ok
  end

  @spec validate_multiple_groups!(Nx.Tensor.t()) :: :ok
  defp validate_multiple_groups!(sensitive_attr) do
    unique_count =
      sensitive_attr
      |> Nx.to_flat_list()
      |> Enum.uniq()
      |> length()

    if unique_count < 2 do
      raise ExFairness.Error, """
      sensitive_attr must contain at least 2 different groups.

      Found only #{unique_count} unique value(s).
      Fairness metrics require comparing multiple groups.
      """
    end

    :ok
  end

  @spec validate_sufficient_samples!(Nx.Tensor.t(), non_neg_integer()) :: :ok
  defp validate_sufficient_samples!(sensitive_attr, min_per_group) do
    group_0_count = Nx.sum(Nx.equal(sensitive_attr, 0)) |> Nx.to_number()
    group_1_count = Nx.sum(Nx.equal(sensitive_attr, 1)) |> Nx.to_number()

    if group_0_count < min_per_group or group_1_count < min_per_group do
      raise ExFairness.Error, """
      Insufficient samples per group for reliable fairness metrics.

      Found:
        Group 0: #{group_0_count} samples
        Group 1: #{group_1_count} samples

      Recommended minimum: #{min_per_group} samples per group.

      Consider:
      - Collecting more data
      - Using bootstrap methods with caution
      - Aggregating smaller groups if appropriate
      """
    end

    :ok
  end
end
