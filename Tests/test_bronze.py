import unittest
import os
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import urllib.parse
from src.ingestion.bronze import api_call

class TestApiCall(unittest.TestCase):

    @patch('src.ingestion.bronze.requests.get')
    @patch('src.ingestion.bronze.os.makedirs')
    @patch('src.ingestion.bronze.os.path.exists')
    @patch('src.ingestion.bronze.open', new_callable=unittest.mock.mock_open)
    def test_api_call(self, mock_open, mock_exists, mock_makedirs, mock_get):
        # Define mock data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"id": "card1"}, {"id": "card2"}],
            "has_more": False
        }
        mock_get.return_value = mock_response
        mock_exists.return_value = False

        root_dir = "./test_dir"
        mode = "predict"
        identifier = "test"

        # Call the function
        predict_days=30
        api_call(root_dir=root_dir, mode=mode, identifier=identifier, predict_days=predict_days)

        # Check if directories and files were created
        mock_makedirs.assert_called_once_with(os.path.join(root_dir, "bronze", mode), exist_ok=True)
        mock_open.assert_called()

        # Check if correct URLs were used
        predict_date_lag = (datetime.now() - timedelta(predict_days)).strftime("%Y-%m-%d")
        base_url = "https://api.scryfall.com/cards/search?q="
        predict_query = urllib.parse.quote(
            f"not:reprint not:digital date>{predict_date_lag}")
        expected_url = f"{base_url}{predict_query}"
        mock_get.assert_called_with(expected_url)

        # Check the file operations
        self.assertTrue(mock_open.called)
        handle = mock_open()
        handle.write.assert_called()


if __name__ == '__main__':
    unittest.main()