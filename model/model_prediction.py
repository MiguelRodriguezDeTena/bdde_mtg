import os
import mlflow
from pyspark.sql import SparkSession, DataFrame

def predict_df(df: DataFrame, config:dict) -> DataFrame:

    df = df.toPandas()

    # mlflow.set_tracking_uri("databricks") #set to databricks
    client = mlflow.MlflowClient()
    last_run = client.get_latest_versions(config["model_name"])[0].run_id
    model_uri = f"runs:/{last_run}/transformers-model"

    pipe = mlflow.transformers.load_model(model_uri)

    target_columns = ['mana_cost','type_line', 'oracle_text', 'bottomright_value','second_mana_cost','second_type_line','second_oracle_text','second_bottomright_value']

    df["results"] = df.apply(lambda x: pipe(" ".join([str(x[col]) for col in target_columns])), axis=1)

    df["label"] = df["results"].apply(lambda x: x[0]["label"])
    df["score"] = df["results"].apply(lambda x: x[0]["score"])

    return df


def write_predict_results(df, config, root_dir, identifier):

    df = predict_df(df, config)

    df = df[["oracle_id", "name", "image_link", "label", "score", "price"]]

    write_file_path = os.path.join(root_dir, f"results")
    write_file_name = f"results_{identifier}.csv"

    if not os.path.exists(write_file_path):
        os.makedirs(write_file_path)

    df.to_csv(f"{write_file_path}/{write_file_name}", index=False, sep=";")