import os
import argparse
import json
from datetime import datetime
from pyspark.sql import DataFrame
import yaml

class PipelineUtils():
    def __init__(self, spark):
        '''
        Initializes the pipeline helper class with the Spark session and parses command line arguments.

        Parameters:
        spark (SparkSession): The active Spark session.
        '''

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--mode")
        self.parser.add_argument("--root_dir")
        self.parser.add_argument("--config_dir")
        self.args = self.parser.parse_args()
        self.mode = self.args.mode
        self.config_dir = self.args.config_dir
        self.root_dir = self.args.root_dir
        self.spark = spark

    def read(self, read_zone:str, identifier:str) -> DataFrame:
        '''
        Reads data from the specified zone (bronze, silver, gold) and returns it as a Spark DataFrame.

        Parameters:
        read_zone (str): The data zone to read from ('bronze', 'silver', or 'gold').
        identifier (str): The unique identifier for the dataset to be read.

        Returns:
        DataFrame: A Spark DataFrame with the loaded data.
        '''

        read_file_path = os.path.join(self.root_dir, read_zone, self.mode)
        read_file_name = f"{self.mode}_{read_zone}_{identifier}"

        if read_zone != "bronze":
            df = self.spark.read.format("csv").option("delimiter", ";").option("header", "true").load(
                f"{read_file_path}/{read_file_name}.csv")
        else:
            df = self.spark.read.json(f"{read_file_path}/{read_file_name}.json")

        print(f"Reading {read_file_path}/{read_file_name}")
        return df

    def manifest(self, overwrite:bool=False) -> str:
        '''
        Manages the manifest file to keep track of the identifier for the current run.

        Parameters:
        overwrite (bool): If True, creates a new manifest file with a new identifier.

        Returns:
        str: The identifier for the current dataset.
        '''

        if overwrite == True:
            manifest_file_path = os.path.join(self.root_dir, "manifest")
            os.makedirs(manifest_file_path, exist_ok=True)
            identifier = datetime.now().strftime("%y%m%d%H%M%S")
            manifest_data = {
                "identifier": identifier
            }
            with open(os.path.join(self.root_dir, "manifest", f"{self.mode}_manifest.json"), "w") as manifest_file:
                json.dump(manifest_data, manifest_file, indent=4)
            print(f"New manifest file overwrite with number {identifier}")
            return identifier
        else:
            with open(os.path.join(self.root_dir, "manifest", f"{self.mode}_manifest.json"), "r") as manifest_file:
                identifier = json.load(manifest_file).get("identifier")
            return identifier

    def write_csv(self, df:DataFrame, write_zone:str, identifier:str) -> None:

        '''
        Writes the DataFrame to a CSV file in the specified write zone (bronze, silver, gold).

        Parameters:
        df (DataFrame): The Spark DataFrame to write to a file.
        write_zone (str): The data zone to write to ('silver' or 'gold').
        identifier (str): The unique identifier for the dataset being written.

        Returns:
        None
        '''


        write_file_path = os.path.join(self.root_dir, write_zone,self.mode)
        write_file_name = f"{self.mode}_{write_zone}_{identifier}.csv"

        if not os.path.exists(write_file_path):
            os.makedirs(write_file_path)

        df.write.option("delimiter", ";").option("header", "true").csv(f"{write_file_path}/{write_file_name}",
                                                                           mode="overwrite")
        print(f"{write_file_path}/{write_file_name} has been written")

    def read_yaml(self):
        with open(self.config_dir, "rb") as file:
            config = yaml.safe_load(file)
        return config
