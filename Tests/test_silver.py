import unittest
from pyspark.sql import SparkSession
from pipeline_utils import PipelineUtils
from ingestion.silver import select_data_from_json, silver_transform

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
        self.predict_df = predict.read(read_zone="bronze",identifier=identifier)
        train = PipelineUtils(self.spark, mode="train", root_dir=root_dir)
        self.train_df = train.read(read_zone="bronze", identifier=identifier)

    def test_select_data_from_json_predict(self):
        df_results = select_data_from_json("predict",self.predict_df)
        self.assertTrue(df_results.count(),"12")
        self.assertIn("name",df_results.columns)
        self.assertIn("image_uris", df_results.columns)
        self.assertTrue(set(['oracle_id','card_face_number','mana_cost','type_line','oracle_text',
                             'loyalty','usd','usd_foil','power','toughness','defense']).issubset(df_results.columns))
    def test_select_data_from_json_train(self):
        df_results = select_data_from_json("train",self.train_df)
        self.assertTrue(df_results.count(),"12")
        self.assertNotIn("name", df_results.columns)
        self.assertNotIn("image_link", df_results.columns)
        self.assertTrue(set(['oracle_id', 'card_face_number', 'mana_cost', 'type_line', 'oracle_text',
                             'loyalty', 'usd', 'usd_foil', 'power', 'toughness', 'defense']).issubset(
            df_results.columns))

    def test_silver_transform_predict(self):
        df_results = silver_transform("predict", self.predict_df)
        self.assertTrue(df_results.count(), "12")
        self.assertIn("name", df_results.columns)
        self.assertIn("image_link", df_results.columns)
        self.assertTrue(set(['oracle_id','card_face_number','mana_cost',
                             'type_line','oracle_text','bottomright_value','price']).issubset(df_results.columns))
    def test_silver_transform_train(self):
        df_results = silver_transform("train", self.train_df)
        self.assertTrue(df_results.count(), "12")
        self.assertNotIn("name", df_results.columns)
        self.assertNotIn("image_link", df_results.columns)
        self.assertTrue(set(['oracle_id', 'card_face_number', 'mana_cost',
                             'type_line', 'oracle_text', 'bottomright_value', 'price']).issubset(df_results.columns))

if __name__ == "__main__":
    unittest.main()
