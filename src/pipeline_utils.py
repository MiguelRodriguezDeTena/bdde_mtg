from datetime import datetime
from pyspark.sql import DataFrame
import yaml
import json

class PipelineUtils():
    def __init__(self, spark, mode=None, root_dir=None ,config_dir=None):
        '''
        Initializes the pipeline helper class with the Spark session and parses command line arguments.

        Parameters:
        spark (SparkSession): The active Spark session.
        mode (str, optional): The mode of operation (e.g., 'train', 'predict'). Defaults to None.
        root_dir (str, optional): The root directory for file reading/writing. Defaults to None.
        config_dir (str, optional): The directory for the configuration YAML file. Defaults to None.
        '''

        self.mode = mode 
        self.root_dir = root_dir 
        self.config_dir = config_dir 
        self.spark = spark

    def read(self, read_zone:str, identifier:str) -> DataFrame:
        '''
        Reads data from the specified zone (bronze, silver, gold) and returns it as a Spark DataFrame.

        Parameters:
        read_zone (str): The data zone to read from (bronze, 'silver', or 'gold').
        identifier (str): The unique identifier for the dataset to be read.

        Returns:
        DataFrame: A Spark DataFrame with the loaded data.
        '''

        read_file_path = f"{self.root_dir}/{read_zone}/{self.mode}"
        read_file_name = f"{self.mode}_{read_zone}_{identifier}"

        if read_zone != "bronze":
            df = self.spark.read.format("csv").option("delimiter", ";").option("header", "true").load(
                f"{read_file_path}/{read_file_name}.csv")
        else:
            df = self.spark.read.format("json").load(f"{read_file_path}/{read_file_name}.json")

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

        if overwrite:
            identifier = datetime.now().strftime("%y%m%d%H%M%S")
            manifest_data = {
                "identifier": identifier
            }

            with open(f"/dbfs/tmp/{self.mode}_manifest.json", "w") as manifest_file:
                json.dump(manifest_data, manifest_file, indent=4)

            print(f"New manifest file overwrite with number {identifier}")
            return identifier
        else:
            with open(f"/dbfs/tmp/{self.mode}_manifest.json", "r") as manifest_file:
                identifier = json.load(manifest_file).get("identifier")
            return identifier

    def write_csv(self, df:DataFrame, write_zone:str, identifier:str) -> None:

        '''
        Writes the DataFrame to a CSV file in the specified write zone (silver, gold), if bronze, reads from the /dbfs/tmp file

        Parameters:
        df (DataFrame): The Spark DataFrame to write to a file.
        write_zone (str): The data zone to write to ('silver' or 'gold').
        identifier (str): The unique identifier for the dataset being written.

        Returns:
        None
        '''


        write_file_path = f"{self.root_dir}/{write_zone}/{self.mode}"
        write_file_name = f"{self.mode}_{write_zone}_{identifier}.csv"

        df.write.option("delimiter", ";").option("header", "true").csv(f"{write_file_path}/{write_file_name}",
                                                                           mode="overwrite")
        print(f"{write_file_path}/{write_file_name} has been written")
    
    def read_yaml(self):

        '''
        Reads a YAML configuration file and returns it as a Python dictionary.

        Returns:
        dict: The parsed YAML configuration file.
        '''

        with open(self.config_dir, "rb") as file:
            config = yaml.safe_load(file)
        return config
