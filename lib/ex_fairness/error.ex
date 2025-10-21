defmodule ExFairness.Error do
  @moduledoc """
  Custom error for ExFairness operations.
  """
  defexception [:message]

  @doc """
  Creates a new ExFairness error with the given message.
  """
  @spec exception(String.t()) :: %__MODULE__{message: String.t()}
  def exception(message) when is_binary(message) do
    %__MODULE__{message: message}
  end
end
