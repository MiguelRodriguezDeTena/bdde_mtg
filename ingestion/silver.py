from pyspark.sql import DataFrame
import pyspark.sql.functions as f

def select_data_from_json(mode:str, bronze: DataFrame) -> DataFrame:
    '''
    Reads card data from a DataFrame and processes it based on the mode.

    Parameters:
    mode (str): The mode of operation, either 'train' or 'predict'. Determines which columns to select.
    bronze (DataFrame): The input DataFrame containing card data.

    Returns:
    DataFrame: A Spark DataFrame containing the processed data, handling both single and double-faced cards.
    '''

    # Define the target columns to select from the data
    target_columns = ["mana_cost", "type_line", "oracle_text", "power", "toughness", "loyalty", "defense"]

    # Add additional columns for visualization if mode is 'predict'
    if mode == "predict":
        target_columns = ["name", "image_uris"] + target_columns

        # Check if the data contains double-faced cards
    if "card_faces" in bronze.columns:
        # Filter for single-faced cards (where card_faces is null)
        single_faced = bronze.filter(f.col("card_faces").isNull())
        single_faced_columns = [col for col in target_columns if col in single_faced.columns]

        # Filter for double-faced cards (where card_faces is not null)
        double_faced = bronze.filter(f.col("card_faces").isNotNull()) \
            .select("oracle_id", f.posexplode("card_faces").alias("card_face_number", "card_faces"),
                    f.col("prices.usd").alias("usd"),
                    f.col("prices.usd_foil").alias("usd_foil"))

        # Define columns to select for double-faced cards
        double_faced_columns = [f.col(f"card_faces.{col}").alias(col) for col in target_columns if
                                col in double_faced.select("card_faces.*").columns]

        # Select data for single-faced cards
        single_faced = single_faced.select("oracle_id",
                                           f.lit("0").alias("card_face_number"),
                                           *single_faced_columns,
                                           f.col("prices.usd").alias("usd"),
                                           f.col("prices.usd_foil").alias("usd_foil"))

        # Select data for double-faced cards
        double_faced = double_faced.select("oracle_id",
                                           "card_face_number",
                                           *double_faced_columns,
                                           "usd",
                                           "usd_foil")

        # Columns present in both single-faced and double-faced DataFrames
        joined_present_columns = [col for col in single_faced.columns if col in double_faced.columns]
        joined_excluded_columns = {col: f.lit(None) for col in target_columns if
                                   col not in single_faced.columns and col not in double_faced.columns}

        # Outer join single-faced and double-faced DataFrames
        silver = single_faced.join(double_faced,
                                   joined_present_columns,
                                   "outer").withColumns(joined_excluded_columns)

    else:
        # Columns present in the DataFrame
        present_columns = [col for col in target_columns if col in bronze.columns]
        # Columns to exclude (fill with None) if they are not present
        excluded_columns = [f.lit(None).alias(col) for col in target_columns if col not in bronze.columns]

        # Select data from bronze DataFrame for single-faced cards
        silver = bronze.select("oracle_id",
                               f.lit("0").alias("card_face_number"),
                               *present_columns,
                               *excluded_columns,
                               f.col("prices.usd").alias("usd"),
                               f.col("prices.usd_foil").alias("usd_foil"))

    # Unpersist the cached DataFrame to free memory
    bronze.unpersist()
    return silver


def silver_transform(mode:str, bronze: DataFrame) -> DataFrame:
    '''
    Apply transformations to the 'silver' DataFrame to prepare it for further use.

    Transformations include:
    - Replacing newlines in 'oracle_text' with whitespace.
    - Removing reminder text within parentheses in 'oracle_text'.
    - Creating a 'bottomright_value' column to represent the card's bottom-right stats (power/toughness, loyalty, defense).
    - Setting a 'price' column based on the 'usd' or 'usd_foil' columns.

    Parameters:
    mode (str): The mode of operation, either 'train' or 'predict'. Determines which columns to select.
    bronze (DataFrame): The DataFrame to be transformed.

    Returns:
    DataFrame: The transformed Spark DataFrame ready for use.
    '''

    bronze = select_data_from_json(mode, bronze)

    # Replace newlines in 'oracle_text' with a space and remove text within parentheses (reminder text)
    # Create 'bottomright_value' based on available attributes: power/toughness, loyalty, or defense
    # Set 'price' column based on 'usd' price or fallback to 'usd_foil' if 'usd' is not available
    silver = bronze.withColumn("oracle_text", f.regexp_replace(f.col("oracle_text"), "[\n]", " ")) \
        .withColumn("oracle_text", f.regexp_replace(f.col("oracle_text"), r"\(.*?\)", "")) \
        .withColumn(
        "bottomright_value",
        f.when(f.col("power").isNotNull() & f.col("toughness").isNotNull(),
               f.concat_ws("/", f.col("power"), f.col("toughness")))
        .when(f.col("loyalty").isNotNull(), f.col("loyalty"))
        .when(f.col("defense").isNotNull(), f.col("defense"))
        .otherwise(None)
    ) \
        .withColumn("price", f.when(f.col("usd").isNotNull(), f.col("usd")).otherwise(f.col("usd_foil")))

    # In predict mode, include 'name' and a small image link for visualization
    if mode == "predict":
        silver = silver.select("oracle_id", "name", f.col("image_uris.small").alias("image_link"),
                               "card_face_number", "mana_cost", "type_line", "oracle_text", "bottomright_value",
                               "price")
    else:
        silver = silver.select("oracle_id", "card_face_number", "mana_cost", "type_line", "oracle_text",
                               "bottomright_value", "price")

    return silver