import pandas as pd
import numpy as np


def parse_column_quantile_intervals(df: pd.DataFrame, bins: int = 10) -> dict:
    """
    Computes quantile-based interval counts for each numeric column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns to be binned.
    bins : int, optional
        Number of quantile bins to compute (default is 10).

    Returns
    -------
    dict
        A dictionary where keys are column names and values are:
            - A dictionary mapping interval strings to value counts if binning succeeds.
            - A string message if the column lacks sufficient variation to compute quantiles.

    Notes
    -----
    - Only numeric columns are processed; others are ignored.
    - NaN values are excluded from the binning operation.
    - Uses unique quantile edges to prevent duplicate bin boundaries.
    """

    result = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Drop NaN values
            non_na_values = df[col].dropna()

            # Create quantile-based bins
            try:
                bin_edges = np.unique(
                    np.quantile(non_na_values, np.linspace(0, 1, bins + 1))
                )

                # Cut based on bin edges
                bin_counts = (
                    pd.cut(
                        non_na_values,
                        bins=bin_edges,
                        include_lowest=True,
                        duplicates="drop",
                    )
                    .value_counts()
                    .sort_index()
                )

                # Store results
                result[col] = {
                    str(interval): count for interval, count in bin_counts.items()
                }
            except ValueError:
                # If all values are the same, quantile splitting will fail
                result[col] = "Not enough variation to split"
        else:
            continue

    return result


def parse_list_quantile_intervals(values: list[float], bins: int = 10) -> dict:
    values = pd.Series(values).dropna()

    try:
        # Create quantile-based bin edges
        bin_edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))

        # Cut values into bins
        bin_counts = (
            pd.cut(values, bins=bin_edges, include_lowest=True, duplicates="drop")
            .value_counts()
            .sort_index()
        )

        return {str(interval): count for interval, count in bin_counts.items()}

    except ValueError:
        # If not enough unique values
        return {"Error": "Not enough variation to split into bins."}
