import os
import mlflow
from pyspark.sql import DataFrame

def predict_df(df: DataFrame, config:dict) -> DataFrame:

    '''
    Applies a pre-trained ML model to predict labels and scores on the input DataFrame.

    Parameters:
    df (DataFrame): The Spark DataFrame to be transformed into a Pandas DataFrame for model inference.
    config (dict): The configuration dictionary containing model information, such as 'model_name'.

    Returns:
    DataFrame: The Pandas DataFrame with prediction results including 'label' and 'score'.
    '''

    df = df.toPandas()

    # mlflow.set_tracking_uri("databricks") #set to databricks
    client = mlflow.MlflowClient()
    last_run = client.get_latest_versions(config["model_name"])[0].run_id
    model_uri = f"runs:/{last_run}/transformers-model"

    pipe = mlflow.transformers.load_model(model_uri)

    target_columns = ['mana_cost','type_line', 'oracle_text', 'bottomright_value','second_mana_cost','second_type_line',
                      'second_oracle_text','second_bottomright_value']

    df["results"] = df.apply(lambda x: pipe(" ".join([str(x[col]) for col in target_columns])), axis=1)

    df["label"] = df["results"].apply(lambda x: x[0]["label"])
    df["score"] = df["results"].apply(lambda x: x[0]["score"])

    return df


def write_predict_results(df:DataFrame, config: dict, root_dir: str, identifier: str) -> None:
    '''
    Writes the prediction results to a CSV file in the specified directory.

    Parameters:
    df (DataFrame): The Spark DataFrame containing the prediction results.
    config (dict): The configuration dictionary passed to the prediction function.
    root_dir (str): The root directory where the result file will be saved.
    identifier (str): A unique identifier to use in the result file's name.

    Returns:
    None
    '''

    df = predict_df(df, config)

    df = df[["oracle_id", "name", "image_link", "label", "score", "price"]]

    write_file_path = os.path.join(root_dir, f"results")
    write_file_name = f"results_{identifier}.csv"

    if not os.path.exists(write_file_path):
        os.makedirs(write_file_path)

    df.to_csv(f"{write_file_path}/{write_file_name}", index=False, sep=";")