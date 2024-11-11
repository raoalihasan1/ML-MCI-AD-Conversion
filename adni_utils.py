import miceforest as mf
import pandas as pd
from typing import Any, Callable

INPUT_DIR = "./raw_data"
OUTPUT_DIR = "./output_data"
DF_COLUMNS = ["RID", "VISCODE2"]
ADNIMERGE_COLUMNS = ["RID", "VISCODE"]


def create_bl_of_col(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Takes the baseline values of a specific column, creates a new
    column in the DataFrame with that column name post-fixed with
    `bl`, and inserts that bl value for each record for each `RID`.

    Args:
        pd.DataFrame: The input DataFrame containing `RID`,
                      `VISCODE`, and the specified column.
        str: The name of the column for which the bl values will be created.

    Returns:
        pd.DataFrame: A new DataFrame with an added column that
                      contains bl values for the specified column.
    """
    bl_name = f"{column_name}_bl"
    bl_values = df[df["VISCODE"] == "bl"][["RID", column_name]]
    df = df.merge(
        bl_values.rename(columns={column_name: bl_name}), on="RID", how="left"
    )
    for rid in df["RID"].unique():
        bl_value = df.loc[df["RID"] == rid, bl_name].iloc[0]
        if pd.notna(bl_value):
            df.loc[df["RID"] == rid, bl_name] = bl_value
    return df


def create_plasma_df(
    column_name: str, df_lst_with_map_f: list[tuple[pd.DataFrame, Callable]]
) -> pd.DataFrame:
    """
    Constructs a single DataFrame from a list of DataFrames
    with a corresponding mapping functions, creating rows
    with a specified column name, the `RID`, and `VISCODE`.

    Args:
        str: The name of the target column to be populated in the new DataFrame.
        list[tuple]:
            A list where each element is a tuple consisting of:
            - A DataFrame containing columns `RID` and `VISCODE2`.
            - A mapping function that transforms each
              row to create values for the target column.

    Returns:
        pd.DataFrame: A new DataFrame containing the columns `RID`, `VISCODE`,
                      and `column_name`, with duplicate and missing values
                      removed. The DataFrame is reset with a continuous index.
    """
    # Create a dataframe with these specific columns
    new_df = pd.DataFrame(columns=["RID", "VISCODE", column_name])
    new_df.drop(new_df.index, inplace=True)

    for df, map_f in df_lst_with_map_f:
        for i in range(df.shape[0]):
            # Catch any errors such as null values
            # or if the passed map function throws
            try:
                row = df.iloc[i]
                data = {
                    "RID": int(row.RID),
                    "VISCODE": row.VISCODE2,
                    # Apply the passed map function on each row
                    column_name: map_f(row),
                }
                new_df.loc[len(new_df.index)] = data
            except:
                pass

    # Drop duplicate, NaN rows and reset the index
    new_df.drop_duplicates(subset=ADNIMERGE_COLUMNS, inplace=True)
    new_df.dropna(inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def df_of_csv(filename: str, input_dir: bool = True) -> pd.DataFrame:
    """
    Reads a CSV file into a DataFrame from the specified directory.

    Args:
        str: The name of the CSV file to read (without the `.csv` extension).
        bool: If True, finds the file in the INPUT_DIR, else from OUTPUT_DIR.

    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """
    return pd.read_csv(f"{INPUT_DIR if input_dir else OUTPUT_DIR}/{filename}.csv")


def filter_n_years_from_bl(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filters rows where `Years_bl` values are greater than or equal to n
    (inclusive), indicating the n-year mark from the baseline. Then,
    it removes duplicates within this subset, keeping only the first
    occurrence of each unique `RID` value.

    Args:
        pd.DataFrame: The DataFrame containing columns `Years_bl` and `RID`.
        int: Years from baseline to filter the data for.

    Returns:
        pd.DataFrame: A DataFrame filtered to rows with `Years_bl` values
                      greater than or equal to n (inclusive), , and with
                      duplicates removed based on `RID` (keeping only the
                      first occurrence of each 'RID').
    """
    # Filter the df for the records that are >= than n years from the bl
    n_years_from_bl = df[(df["Years_bl"] >= n)]
    # Return only the first record for each RID of the filtered df
    return n_years_from_bl.drop_duplicates(subset="RID", keep="first")


def map_col_to_num(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Maps unique categorical values in a specified column of a DataFrame to integers.
    This is useful for encoding categorical data as numerical values.

    Args:
        pd.DataFrame: The input DataFrame containing the column to be mapped.
        str: The name of the column whose values should be mapped to integers.

    Returns:
        pd.DataFrame: A DataFrame with the specified column's values mapped to integers.
    """
    vals = df[column_name].unique()
    df[column_name] = df[column_name].replace(vals, list(range(len(vals))))
    return df


def mice_forest_data_imputation(df: pd.DataFrame, iterations: int = 5) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame using the MICE (Multiple Imputation by
    Chained Equations) algorithm with random forests from the `miceforest` library.

    Args:
        pd.DataFrame: The input DataFrame containing missing values to be imputed.
        int: The number of MICE iterations to perform. (Default is 5)

    Returns:
        pd.DataFrame: A DataFrame with imputed values, containing no missing values.
    """
    kernel = mf.ImputationKernel(
        df.reset_index(drop=True), num_datasets=4, random_state=0
    )
    kernel.mice(iterations=iterations)
    return kernel.complete_data()


def print_no_of_rows_removed(df: pd.DataFrame, df1: pd.DataFrame) -> None:
    """
    Print the number & percentage of rows removed after filtering a DataFrame.

    Args:
        pd.DataFrame: The DataFrame before filtering.
        pd.DataFrame: The DataFrame after filtering.

    Returns:
        None
    """
    diff = df.shape[0] - df1.shape[0]
    percent = round((diff / df.shape[0]) * 100, 2)
    print(f"{diff} rows removed through data processing ({percent}% removed)")


def remove_float_values(n: Any) -> bool:
    """
    Checks if a value cannot be converted to a float, returning True
    if the conversion fails (indicating it's not a float-compatible value),
    and False if it succeeds (indicating it's float-compatible).

    Args:
        Any: The value to check for float compatibility.

    Returns:
        bool: True if the value cannot be converted to a float, otherwise False.
    """
    try:
        float(n)
    except ValueError:
        return True
    return False


def remove_rows_not_in_adnimerge(
    adnimerge_df: pd.DataFrame, df: pd.DataFrame, print_statistics: bool = True
) -> pd.DataFrame:
    """
    Filters a DataFrame by retaining only the rows with
    matching `VISCODE2` and `RID` in the ADNIMERGE dataset.

    Args:
        pd.DataFrame: The ADNIMERGE DataFrame.
        pd.DataFrame: The input DataFrame to be filtered.
        bool: Controls whether to print statistics about the filtering.

    Returns:
        pd.DataFrame: A filtered DataFrame with rows matching only
                      those also present in the ADNIMERGE dataset.
    """
    # Some df have the column labelled as VISCODE,
    # so to standardise it, we rename it to VISCODE2
    if "VISCODE" in df.columns and "VISCODE2" not in df.columns:
        df.rename(columns={"VISCODE": "VISCODE2"}, inplace=True)

    # Replace sc (screening) with bl (baseline) in the plasma df
    # since ADNIMERGE uses bl hence reduce noise in filtering
    df.loc[df["VISCODE2"] == "sc", "VISCODE2"] = "bl"

    filtered_df = df.merge(
        adnimerge_df,
        left_on=DF_COLUMNS,
        right_on=ADNIMERGE_COLUMNS,
        how="inner",
    )

    # Print some statistics about the filtering
    print_no_of_rows_removed(df, filtered_df) if print_statistics else None
    return filtered_df


def replace_nan_with_surrounding_matching_val(
    df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Fills NaN values in a specified column by matching values from adjacent rows
    when certain conditions are met.

    This function iterates through each row in the DataFrame, looking for
    NaN values in the specified column. If a NaN is found in a row where:
    - The `RID` value is the same in the row above, current, and the row below.
    - The values in the specified column are identical in the rows immediately
      above and below the NaN row.

    If found, replace the NaN value with the matching surrounding value.

    Args:
        pd.DataFrame: The input DataFrame containing the data with NaN values
                      to be replaced.
        str: The name of the column where NaN values should be filled.

    Returns:
        pd.DataFrame: The modified DataFrame with NaN values in the specified
                      columns replaced, if they met the above criteria.
    """
    for i in range(1, df.shape[0] - 1):
        # Check if the value in the specified column is NaN
        if pd.isna(df.loc[i, column_name]):
            # Check if the RID above and below match the current RID and
            # that the values in the above and below column are the same
            if (
                df.loc[i - 1, "RID"] == df.loc[i, "RID"] == df.loc[i + 1, "RID"]
                and df.loc[i - 1, column_name] == df.loc[i + 1, column_name]
            ):
                # Replace NaN with the above and below matching value
                df.loc[i, column_name] = df.loc[i - 1, column_name]
    return df


def statistics_df_of_df(
    df: pd.DataFrame, exclusion_cols: list[str] = []
) -> pd.DataFrame:
    """
    Computes common statistical measures for the numeric columns in a DataFrame
    and returns a new DataFrame containing those statistics.

    This function calculates the following statistics for each numeric column:
    - Mean
    - Count (Sample Size)
    - Standard Deviation
    - Median
    - Range
    - Interquartile Range (IQR)
    - Skewness
    - Minimum Value
    - Maximum Value

    It drops any NaN values before performing the calculations, and
    optionally, specific columns can be excluded from the analysis.

    Args:
        pd.DataFrame: The input DataFrame containing the numerical data.

        list[str]: A list of column names to exclude from
                   the statistics calculation. (Default is [])

    Returns:
        pd.DataFrame: A DataFrame containing the calculated
                      statistics for each numerical column.
    """
    filtered_df = df.drop(columns=exclusion_cols)
    stats_df = pd.DataFrame(
        columns=[
            "Column",
            "Count",
            "Mean",
            "Standard Deviation",
            "Median",
            "Range",
            "IQR",
            "Skew",
            "Min",
            "Max",
        ]
    )
    for column in filtered_df.columns:
        # Drop NaN values for the specific column
        col_data = filtered_df[column].dropna().astype(float)
        stats_df.loc[stats_df.shape[0]] = {
            "Column": column,
            "Count": len(col_data),
            "Mean": col_data.mean().round(3),
            "Standard Deviation": col_data.std().round(3),
            "Median": col_data.median().round(3),
            "Range": (col_data.max() - col_data.min()).round(3),
            "IQR": (col_data.quantile(0.75) - col_data.quantile(0.25)).round(3),
            "Skew": col_data.skew().round(3),
            "Min": col_data.min().round(3),
            "Max": col_data.max().round(3),
        }
    return stats_df
