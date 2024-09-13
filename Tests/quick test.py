from pipeline_utils import PipelineUtils
from ingestion.bronze import api_call
from ingestion.silver import silver_transform
from ingestion.gold import gold_transform
from model.model_training import train_model
from model.model_prediction import write_predict_results
from pyspark.sql import SparkSession
import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # init sparksession, not neccesary on databricks
    spark = SparkSession.builder.config("spark.executor.memory", "4g").config("spark.driver.memory", "2g").getOrCreate()

    plu = PipelineUtils(spark)
    mode = plu.mode
    root_dir = plu.root_dir
    config = plu.read_yaml()
    identifier = plu.manifest()
    api_call(root_dir, mode, identifier)
    bronze = plu.read("bronze", identifier)
    silver = silver_transform(mode, bronze)
    plu.write_csv(silver, "silver", identifier)
    silver = plu.read("silver", identifier)
    gold = gold_transform(mode, silver)
    plu.write_csv(gold, "gold", identifier)
    gold = plu.read("gold", identifier)
    # train_model(gold,config)
    write_predict_results(gold,config,root_dir,identifier)






