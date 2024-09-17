import unittest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
import pandas as pd
from src.model.model_prediction import predict_df
from src.pipeline_utils import PipelineUtils

class TestPredictDf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize SparkSession for tests
        cls.spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Stop SparkSession after tests
        cls.spark.stop()

    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.search_runs')
    @patch('mlflow.transformers.load_model')
    def test_predict_df(self, mock_load_model, mock_search_runs, mock_get_experiment_by_name):
        # Mock the MLflow experiment, run search, and model loading behavior
        mock_get_experiment_by_name.return_value = MagicMock(experiment_id='test-experiment-id')
        mock_search_runs.return_value = pd.DataFrame({"run_id": ["run-123"]})

        # Mock the pipeline model
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "Above_3", "score": 0.85}]
        mock_load_model.return_value = mock_pipe

        plu = PipelineUtils(self.spark, mode="predict", root_dir="test_dir", config_dir="test_dir/test_config.yaml")

        df = plu.read("gold", "test")

        # Configuration for the model
        config = {
            'experiment_path': '/experiments/path',
            'model_name': 'distilbert'
        }

        # Call the function to predict
        result_df = predict_df(df, config)

        # Check that the result is a Pandas DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)

        # Check that the result DataFrame contains the 'label' and 'score' columns
        self.assertIn("label", result_df.columns)
        self.assertIn("score", result_df.columns)

        # Check that the predictions are as expected
        self.assertEqual(result_df["label"].iloc[0], "Above_3")
        self.assertAlmostEqual(result_df["score"].iloc[0], 0.85)


if __name__ == "__main__":
    unittest.main()
