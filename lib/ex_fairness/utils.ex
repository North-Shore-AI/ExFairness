defmodule ExFairness.Utils do
  @moduledoc """
  Utility functions for ExFairness computations.

  Provides core numerical operations for fairness metrics using Nx tensors.
  All functions in this module are optimized for GPU acceleration via Nx.Defn.
  """

  import Nx.Defn

  @doc """
  Computes the positive prediction rate for a masked subset of predictions.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `mask` - Binary mask tensor indicating which samples to include

  ## Returns

  A scalar tensor containing the positive prediction rate (between 0 and 1).

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1])
      iex> mask = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0])
      iex> rate = ExFairness.Utils.positive_rate(predictions, mask)
      iex> Nx.to_number(rate)
      0.75

  """
  @spec positive_rate(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn positive_rate(predictions, mask) do
    # Apply mask to predictions
    masked_preds = Nx.select(mask, predictions, 0)
    # Count samples in mask
    count = Nx.sum(mask)

    # Compute rate
    Nx.sum(masked_preds) / count
  end

  @doc """
  Creates a binary mask for a specific group value.

  ## Parameters

    * `sensitive_attr` - Tensor of sensitive attribute values
    * `group_value` - The group value to create a mask for (typically 0 or 1)

  ## Returns

  A binary mask tensor where 1 indicates membership in the specified group.

  ## Examples

      iex> sensitive_attr = Nx.tensor([0, 0, 1, 1, 0, 1])
      iex> mask = ExFairness.Utils.create_group_mask(sensitive_attr, 0)
      iex> Nx.to_flat_list(mask)
      [1, 1, 0, 0, 1, 0]

  """
  @spec create_group_mask(Nx.Tensor.t(), number()) :: Nx.Tensor.t()
  defn create_group_mask(sensitive_attr, group_value) do
    Nx.equal(sensitive_attr, group_value)
  end

  @doc """
  Counts the number of samples in a specific group.

  ## Parameters

    * `sensitive_attr` - Tensor of sensitive attribute values
    * `group_value` - The group value to count (typically 0 or 1)

  ## Returns

  A scalar tensor containing the count of samples in the group.

  ## Examples

      iex> sensitive_attr = Nx.tensor([0, 0, 1, 1, 0, 1])
      iex> count = ExFairness.Utils.group_count(sensitive_attr, 0)
      iex> Nx.to_number(count)
      3

  """
  @spec group_count(Nx.Tensor.t(), number()) :: Nx.Tensor.t()
  defn group_count(sensitive_attr, group_value) do
    mask = create_group_mask(sensitive_attr, group_value)
    Nx.sum(mask)
  end

  @doc """
  Computes positive prediction rates for both groups.

  ## Parameters

    * `predictions` - Binary predictions tensor
    * `sensitive_attr` - Binary sensitive attribute tensor

  ## Returns

  A tuple `{rate_group_0, rate_group_1}` of positive prediction rates.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 1, 0, 1])
      iex> sensitive_attr = Nx.tensor([0, 0, 0, 1, 1, 1])
      iex> {rate_0, rate_1} = ExFairness.Utils.group_positive_rates(predictions, sensitive_attr)
      iex> r0 = Nx.to_number(rate_0) |> Float.round(2)
      iex> r1 = Nx.to_number(rate_1) |> Float.round(2)
      iex> {r0, r1}
      {0.67, 0.67}

  """
  @spec group_positive_rates(Nx.Tensor.t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  defn group_positive_rates(predictions, sensitive_attr) do
    mask_0 = create_group_mask(sensitive_attr, 0)
    mask_1 = create_group_mask(sensitive_attr, 1)

    rate_0 = positive_rate(predictions, mask_0)
    rate_1 = positive_rate(predictions, mask_1)

    {rate_0, rate_1}
  end
end
