import unittest
from pyspark.sql import SparkSession
from unittest.mock import patch, MagicMock
from src.pipeline_utils import PipelineUtils
import os

class TestCardProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initialize the PySpark session
        """
        cls.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("UnitTest") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        """
        Stop the PySpark session
        """
        cls.spark.stop()

    def setUp(self):
        """
        Prepare the test DataFrames for each test
        """
        self.identifier = "test"
        self.root_dir = "./test_dir"

    def test_read_predict_bronze(self):
        read_zone = "bronze"
        plu = PipelineUtils(self.spark, mode="predict", root_dir=self.root_dir)
        results = plu.read(read_zone,self.identifier)
        self.assertTrue(results.count(), "10")
    def test_read_predict_silver(self):
        read_zone = "silver"
        plu = PipelineUtils(self.spark, mode="predict", root_dir=self.root_dir)
        df_results = plu.read(read_zone, self.identifier)

        self.assertTrue(df_results.count(), "12")
        self.assertIn("name", df_results.columns)
        self.assertIn("image_link", df_results.columns)
        self.assertTrue(set(['oracle_id', 'card_face_number', 'mana_cost',
                             'type_line', 'oracle_text', 'bottomright_value', 'price']).issubset(df_results.columns))
    def test_read_predict_gold(self):
        read_zone = "gold"
        plu = PipelineUtils(self.spark, mode="predict", root_dir=self.root_dir)
        df_results = plu.read(read_zone, self.identifier)

        self.assertTrue(df_results.count(), "10")
        self.assertIn("name", df_results.columns)
        self.assertIn("image_link", df_results.columns)
        self.assertIn("second_name", df_results.columns)
        self.assertIn("second_image_link", df_results.columns)
        self.assertTrue(set(['oracle_id', 'mana_cost', 'type_line',
                             'oracle_text', 'bottomright_value', 'second_mana_cost', 'second_type_line',
                             'second_oracle_text', 'second_bottomright_value', 'price']).issubset(
            df_results.columns))
        self.assertTrue(df_results.collect()[0].bottomright_value, "[Null]")
    def test_read_train_bronze(self):
        read_zone = "bronze"
        plu = PipelineUtils(self.spark, mode="train", root_dir=self.root_dir)
        results = plu.read(read_zone, self.identifier)
        self.assertTrue(results.count(), "10")
    def test_read_train_silver(self):
        read_zone = "silver"
        plu = PipelineUtils(self.spark, mode="train", root_dir=self.root_dir)
        df_results = plu.read(read_zone, self.identifier)

        self.assertTrue(df_results.count(), "12")
        self.assertNotIn("name", df_results.columns)
        self.assertNotIn("image_link", df_results.columns)
        self.assertTrue(set(['oracle_id', 'card_face_number', 'mana_cost',
                             'type_line', 'oracle_text', 'bottomright_value', 'price']).issubset(df_results.columns))
    def test_read_train_gold(self):
        read_zone = "gold"
        plu = PipelineUtils(self.spark, mode="train", root_dir=self.root_dir)
        df_results = plu.read(read_zone, self.identifier)

        self.assertTrue(df_results.count(), "10")
        self.assertNotIn("name", df_results.columns)
        self.assertNotIn("image_link", df_results.columns)
        self.assertNotIn("second_name", df_results.columns)
        self.assertNotIn("second_image_link", df_results.columns)
        self.assertTrue(set(['oracle_id', 'mana_cost', 'type_line',
                             'oracle_text', 'bottomright_value', 'second_mana_cost', 'second_type_line',
                             'second_oracle_text', 'second_bottomright_value', 'price']).issubset(
            df_results.columns))
        self.assertTrue(df_results.collect()[0].bottomright_value, "[Null]")

    def test_write_csv_predict_silver(self):

        mode = "predict"
        write_zone = "silver"

        mock_df = MagicMock()
        mock_write = mock_df.write.option.return_value
        mock_write.option.return_value = mock_write
        mock_write.csv = MagicMock()

        spark = MagicMock()
        pipeline_utils = PipelineUtils(spark, mode=mode, root_dir="./test_dir")

        pipeline_utils.write_csv(mock_df, write_zone, "test")

        # Check if the DataFrame's write.csv method was called with correct parameters
        mock_write.csv.assert_called_once_with(
            f"./test_dir/{write_zone}/{mode}/{mode}_{write_zone}_test.csv", mode="overwrite")


    def test_write_csv_predict_gold(self):
        mode = "predict"
        write_zone = "gold"

        mock_df = MagicMock()
        mock_write = mock_df.write.option.return_value
        mock_write.option.return_value = mock_write
        mock_write.csv = MagicMock()

        spark = MagicMock()
        pipeline_utils = PipelineUtils(spark, mode=mode, root_dir="./test_dir")

        pipeline_utils.write_csv(mock_df, write_zone, "test")

        # Check if the DataFrame's write.csv method was called with correct parameters
        mock_write.csv.assert_called_once_with(
            f"./test_dir/{write_zone}/{mode}/{mode}_{write_zone}_test.csv", mode="overwrite")
    def test_write_csv_train_silver(self):
        mode = "train"
        write_zone = "silver"

        mock_df = MagicMock()
        mock_write = mock_df.write.option.return_value
        mock_write.option.return_value = mock_write
        mock_write.csv = MagicMock()

        spark = MagicMock()
        pipeline_utils = PipelineUtils(spark, mode=mode, root_dir="./test_dir")

        pipeline_utils.write_csv(mock_df, write_zone, "test")

        # Check if the DataFrame's write.csv method was called with correct parameters
        mock_write.csv.assert_called_once_with(
            f"./test_dir/{write_zone}/{mode}/{mode}_{write_zone}_test.csv", mode="overwrite")
    def test_write_csv_train_gold(self):
        mode = "train"
        write_zone = "gold"

        mock_df = MagicMock()
        mock_write = mock_df.write.option.return_value
        mock_write.option.return_value = mock_write
        mock_write.csv = MagicMock()

        spark = MagicMock()
        pipeline_utils = PipelineUtils(spark, mode=mode, root_dir="./test_dir")

        pipeline_utils.write_csv(mock_df, write_zone, "test")

        # Check if the DataFrame's write.csv method was called with correct parameters
        mock_write.csv.assert_called_once_with(
            f"./test_dir/{write_zone}/{mode}/{mode}_{write_zone}_test.csv", mode="overwrite")

    def test_read_yaml(self):
        spark = MagicMock()
        config = PipelineUtils(spark,config_dir="test_dir/test_config.yaml").read_yaml()
        self.assertTrue(config["readable"], "yes")

    def test_manifest(self):
        spark = MagicMock()
        train_identifier = PipelineUtils(spark, mode="train", root_dir="test_dir").manifest()
        predict_identifier = PipelineUtils(spark, mode="predict", root_dir="test_dir").manifest()
        self.assertTrue(train_identifier, "test")
        self.assertTrue(predict_identifier, "test")

if __name__ == "__main__":
    unittest.main()

