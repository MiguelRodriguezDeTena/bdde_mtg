import unittest
from pyspark.sql import SparkSession
from src.pipeline_utils import PipelineUtils
from src.ingestion.gold import gold_transform

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
        identifier = "test"
        root_dir = "./test_dir"
        predict = PipelineUtils(self.spark, mode="predict", root_dir=root_dir)
        self.predict_df = predict.read(read_zone="silver",identifier=identifier)
        train = PipelineUtils(self.spark, mode="train", root_dir=root_dir)
        self.train_df = train.read(read_zone="silver", identifier=identifier)

    def test_gold_transform_predict(self):
        df_results = gold_transform("predict", self.predict_df)
        self.assertTrue(df_results.count(), "10")
        self.assertIn("name", df_results.columns)
        self.assertIn("image_link", df_results.columns)
        self.assertIn("second_name", df_results.columns)
        self.assertIn("second_image_link", df_results.columns)
        self.assertTrue(set(['oracle_id','mana_cost','type_line',
                             'oracle_text','bottomright_value','second_mana_cost','second_type_line',
                             'second_oracle_text','second_bottomright_value','price']).issubset(
            df_results.columns))
        self.assertTrue(df_results.collect()[0].bottomright_value, "[Null]")

    def test_gold_transform_train(self):
        df_results = gold_transform("train", self.train_df)
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

if __name__ == "__main__":
    unittest.main()
