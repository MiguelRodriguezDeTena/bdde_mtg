from pyspark.sql import DataFrame
import pyspark.sql.functions as f

def gold_transform(mode:str, silver: DataFrame) -> DataFrame:
    '''
    Transforms the 'silver' DataFrame into a 'gold' DataFrame, ready for machine learning tasks.

    The transformation includes handling double-faced cards by pivoting data and creating columns for both card faces.
    It also formats the DataFrame differently based on the mode ('train' or 'predict') and handles missing values.

    Parameters:
    mode (str): The mode of operation, either 'train' or 'predict'. Determines additional columns to be included.
    silver (DataFrame): The Spark DataFrame containing silver-level data.

    Returns:
    DataFrame: A transformed Spark DataFrame suitable for machine learning tasks.
    '''

    # Define the target columns to be considered
    target_columns = ["mana_cost", "type_line", "oracle_text", "bottomright_value"]

    # Add extra columns for prediction mode
    if mode == "predict":
        target_columns = ["name",
                          "image_link"] + target_columns  # add extra columns for later use if mode is predict

    # Check if there are two distinct card faces
    if len(silver.select("card_face_number").distinct().collect()) == 2:

        # Columns that are present in the data for both card faces
        present_columns = [f.first(f"{col}").cast("string").alias(f"{col}") for col in target_columns]

        # Pivot data by card_face_number and group by oracle_id
        gold = silver.groupBy("oracle_id").pivot("card_face_number").agg(
            *present_columns,
            f.first("price").cast("string").alias("price"))

        # Columns for card face 0
        grouped_0_columns = [f.col(f"0_{col}").alias(f"{col}")
                             for col in target_columns
                             if f"0_{col}" in gold.columns]

        # Columns for card face 1 (renamed with 'second_' prefix)
        grouped_present_1_columns = [f.col(f"1_{col}").alias(f"second_{col}")
                                     for col in target_columns
                                     if f"1_{col}" in gold.columns]

        # Missing columns for card face 1 (filled with None)
        grouped_missing_1_columns = [f.lit(None).cast("string").alias(f"second_{col}")
                                     for col in target_columns
                                     if f"1_{col}" not in gold.columns]

        # Select and combine columns into the final gold DataFrame
        gold = gold.select(
            "oracle_id",
            *grouped_0_columns,
            *grouped_present_1_columns,
            *grouped_missing_1_columns,
            f.col("0_price").alias("price"))

    else:
        # If only one card face is present, add missing columns for the second face
        missing_columns = [f.lit(None).cast("string").alias(f"second_{col}") for col in
                           target_columns]  # add missing columns

        gold = silver.select(
            "oracle_id",
            *target_columns,
            *missing_columns,
            "price")

    # Replace null values with "[Null]" for better handling in text classification models
    gold = gold.fillna({
        "mana_cost": "[Null]",
        "type_line": "[Null]",
        "oracle_text": "[Null]",
        "bottomright_value": "[Null]",
        "second_mana_cost": "[Null]",
        "second_type_line": "[Null]",
        "second_oracle_text": "[Null]",
        "second_bottomright_value": "[Null]"
    })

    return gold