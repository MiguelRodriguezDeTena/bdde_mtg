import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from datasets import Dataset
from src.pipeline_utils import PipelineUtils
from src.model.model_training import df_to_dataset


class TestDfToDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize SparkSession for tests
        cls.spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Stop SparkSession after tests
        cls.spark.stop()

    def test_df_to_dataset(self):

        plu = PipelineUtils(self.spark, mode="train", root_dir="test_dir", config_dir="test_dir/test_config.yaml")

        df = plu.read("gold","test")
        config = plu.read_yaml()

        # Run the df_to_dataset function
        result_dataset = df_to_dataset(df)

        # Assert the result is a Hugging Face Dataset
        self.assertIsInstance(result_dataset, Dataset)

        # Assert that the resulting dataset has the correct column names
        self.assertListEqual(result_dataset.column_names, ["gameplay_text", "price"])

        # Assert that the label (price) column has the expected class labels
        self.assertEqual(result_dataset.features['price'].names, ["Below_3", "Above_3"])

        # Assert the number of rows matches the input DataFrame
        self.assertEqual(len(result_dataset), 10)

        # Check if gameplay_text is correctly concatenated (simple check on first row)
        expected_gameplay_text = ("{3}{G} Creature — Treefolk [Null] 3/4 [Null] [Null] [Null] [Null]")
        self.assertEqual(result_dataset[4]["gameplay_text"], expected_gameplay_text)


if __name__ == "__main__":
    unittest.main()
