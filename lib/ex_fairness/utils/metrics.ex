defmodule ExFairness.Utils.Metrics do
  @moduledoc """
  Utility functions for computing classification metrics.

  Provides confusion matrix computation and derived metrics like TPR, FPR, and PPV.
  All functions are GPU-accelerated via Nx.Defn.
  """

  import Nx.Defn

  @type confusion_matrix :: %{
          tp: Nx.Tensor.t(),
          fp: Nx.Tensor.t(),
          tn: Nx.Tensor.t(),
          fn: Nx.Tensor.t()
        }

  @doc """
  Computes confusion matrix for masked subset of predictions and labels.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `mask` - Binary mask tensor indicating which samples to include

  ## Returns

  A map containing:
    * `:tp` - True positives count
    * `:fp` - False positives count
    * `:tn` - True negatives count
    * `:fn` - False negatives count

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 0])
      iex> labels = Nx.tensor([1, 0, 0, 0])
      iex> mask = Nx.tensor([1, 1, 1, 1])
      iex> cm = ExFairness.Utils.Metrics.confusion_matrix(predictions, labels, mask)
      iex> {Nx.to_number(cm.tp), Nx.to_number(cm.fp), Nx.to_number(cm.tn), Nx.to_number(cm.fn)}
      {1, 1, 2, 0}

  """
  @spec confusion_matrix(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: confusion_matrix()
  defn confusion_matrix(predictions, labels, mask) do
    # Compute boolean conditions
    pred_pos = Nx.equal(predictions, 1)
    pred_neg = Nx.equal(predictions, 0)
    label_pos = Nx.equal(labels, 1)
    label_neg = Nx.equal(labels, 0)

    # Compute counts with mask
    tp = Nx.sum(Nx.select(mask, Nx.logical_and(pred_pos, label_pos), 0))
    fp = Nx.sum(Nx.select(mask, Nx.logical_and(pred_pos, label_neg), 0))
    tn = Nx.sum(Nx.select(mask, Nx.logical_and(pred_neg, label_neg), 0))
    fn_count = Nx.sum(Nx.select(mask, Nx.logical_and(pred_neg, label_pos), 0))

    %{tp: tp, fp: fp, tn: tn, fn: fn_count}
  end

  @doc """
  Computes True Positive Rate (TPR) / Recall / Sensitivity.

  TPR = TP / (TP + FN)

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `mask` - Binary mask tensor indicating which samples to include

  ## Returns

  A scalar tensor containing the TPR.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 1])
      iex> labels = Nx.tensor([1, 0, 1, 0])
      iex> mask = Nx.tensor([1, 1, 1, 1])
      iex> tpr = ExFairness.Utils.Metrics.true_positive_rate(predictions, labels, mask)
      iex> Nx.to_number(tpr)
      1.0

  """
  @spec true_positive_rate(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn true_positive_rate(predictions, labels, mask) do
    cm = confusion_matrix(predictions, labels, mask)
    denominator = cm.tp + cm.fn

    # Handle division by zero: if no positive labels, return 0
    Nx.select(Nx.equal(denominator, 0), 0.0, cm.tp / denominator)
  end

  @doc """
  Computes False Positive Rate (FPR).

  FPR = FP / (FP + TN)

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `mask` - Binary mask tensor indicating which samples to include

  ## Returns

  A scalar tensor containing the FPR.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 1])
      iex> labels = Nx.tensor([1, 0, 1, 0])
      iex> mask = Nx.tensor([1, 1, 1, 1])
      iex> fpr = ExFairness.Utils.Metrics.false_positive_rate(predictions, labels, mask)
      iex> Nx.to_number(fpr)
      0.5

  """
  @spec false_positive_rate(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn false_positive_rate(predictions, labels, mask) do
    cm = confusion_matrix(predictions, labels, mask)
    denominator = cm.fp + cm.tn

    # Handle division by zero: if no negative labels, return 0
    Nx.select(Nx.equal(denominator, 0), 0.0, cm.fp / denominator)
  end

  @doc """
  Computes Positive Predictive Value (PPV) / Precision.

  PPV = TP / (TP + FP)

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `labels` - Binary labels tensor (0 or 1)
    * `mask` - Binary mask tensor indicating which samples to include

  ## Returns

  A scalar tensor containing the PPV.

  ## Examples

      iex> predictions = Nx.tensor([1, 0, 1, 1])
      iex> labels = Nx.tensor([1, 0, 1, 0])
      iex> mask = Nx.tensor([1, 1, 1, 1])
      iex> ppv = ExFairness.Utils.Metrics.positive_predictive_value(predictions, labels, mask)
      iex> Float.round(Nx.to_number(ppv), 2)
      0.67

  """
  @spec positive_predictive_value(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn positive_predictive_value(predictions, labels, mask) do
    cm = confusion_matrix(predictions, labels, mask)
    denominator = cm.tp + cm.fp

    # Handle division by zero: if no positive predictions, return 0
    Nx.select(Nx.equal(denominator, 0), 0.0, cm.tp / denominator)
  end
end
