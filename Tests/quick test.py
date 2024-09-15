from src.pipeline_utils import PipelineUtils
from src.ingestion.bronze import api_call
from src.ingestion.silver import silver_transform, select_data_from_json
from src.ingestion.gold import gold_transform
from src.model.model_training import train_model
from src.model.model_prediction import write_predict_results
from pyspark.sql import SparkSession



if __name__ == "__main__":
    # init sparksession, not neccesary on databricks
    spark = SparkSession.builder.getOrCreate()

    cma = '--mode "predict" --root_dir "c://users/mykel/bdde_mtg" --config_dir "config.yaml"'

    plu = PipelineUtils(spark, mode="predict", root_dir="c://users/mykel/bdde_mtg")#, config_dir="config.yaml")
    mode = plu.mode
    root_dir = plu.root_dir
    # config = plu.read_yaml()
    identifier = plu.manifest(overwrite=True)
    api_call(spark,root_dir, mode, identifier)
    # bronze = plu.read("bronze", identifier)
    # silver = select_data_from_json(mode, bronze)
    # silver = silver_transform(mode, bronze)
    # plu.write_csv(silver, "silver", identifier)
    # silver = plu.read("silver", identifier)
    # gold = gold_transform(mode, silver)
    # plu.write_csv(gold, "gold", identifier)
    # gold = plu.read("gold", identifier)
    # # train_model(gold,config)
    # write_predict_results(gold,config,root_dir,identifier)
    print("stop")





