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

    #get recent model version's uri
    experiment_id = mlflow.get_experiment_by_name(f"{config['experiment_path']}/{config['model_name']}").experiment_id
    last_run = mlflow.search_runs(experiment_ids=[experiment_id])["run_id"].tolist()[-1]
    model_uri = f"runs:/{last_run}/transformers-model"

    #initialize the pipeline
    pipe = mlflow.transformers.load_model(model_uri)


    target_columns = ['mana_cost','type_line', 'oracle_text', 'bottomright_value','second_mana_cost','second_type_line',
                      'second_oracle_text','second_bottomright_value']

    df["results"] = df.apply(lambda x: pipe(" ".join([str(x[col]) for col in target_columns])), axis=1)

    df["label"] = df["results"].apply(lambda x: x[0]["label"])
    df["score"] = df["results"].apply(lambda x: x[0]["score"])

    return df


def write_predict_results(spark, df:DataFrame, config: dict, root_dir: str, identifier: str) -> None:
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

    #select rows
    df = df[["oracle_id", "name", "image_link", "label", "score", "price"]]

    file_name = f"results_{identifier}.csv"

    #store it as a temp file in dbfs
    temp_file_path = f"/dbfs/tmp/{file_name}"
    df.to_csv(temp_file_path, index=False, sep=";")

    #read temp and write in catalog for PowerBi to connect to.
    df = spark.read.option("delimiter", ";").option("header", "true").csv(f"/tmp/{file_name}")
    df.write.format("delta").mode("overwrite").saveAsTable("results")
