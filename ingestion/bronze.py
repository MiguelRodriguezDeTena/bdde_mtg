import os
import json
from datetime import datetime, timedelta
import urllib.parse
import requests
import time

def api_call(root_dir:str, mode:str, identifier:str, predict_days:int=30, train_days:int=365) -> None:
    '''
    Make an API call to Scryfall, handle pagination, and save the results to a file.

    Parameters:
    root_dir (str): The root directory where the results will be saved.
    mode (str): The mode of operation, either 'train' or 'predict'. Determines the API query and file naming.
    identifier (str): A unique identifier for the file name, e.g., a timestamp or ID.
    predict_days (int, optional): The number of days prior to the current date to use for prediction queries. Default is 30 days.
    train_days (int, optional): The number of days prior to the current date to use for training queries. Default is 365 days.
    '''

    file_path = os.path.join(root_dir, "bronze", mode)
    file_name = f"{mode}_bronze_{identifier}"

    # generates an url depending of mode
    predict_date_lag = (datetime.now() - timedelta(predict_days)).strftime("%Y-%m-%d")
    train_date_lag = (datetime.now() - timedelta(train_days)).strftime("%Y-%m-%d")
    base_url = "https://api.scryfall.com/cards/search?q="
    predict_query = urllib.parse.quote(
        f"not:reprint not:digital date>{predict_date_lag}")  # example "2024-07-18", it's a month prior current date
    train_query = urllib.parse.quote(
        f"legal:legacy -(year<2011 and (prints>1)) year>=2011 ((cheapest:usd and usd>0) or (usd>0)) date<{train_date_lag}")
    test_query = urllib.parse.quote("'rowan'") #for testing purposes
    if mode:
        if mode == "train":
            url = f"{base_url}{train_query}"
        elif mode == "predict":
            url = f"{base_url}{predict_query}"
        elif mode == "test":
            url = f"{base_url}{test_query}"
        else:
            raise Exception("Invalid mode provided.")

    file_index = 0
    start = time.time()

    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            break

        data = response.json()
        if data.get("object") == "error":
            print(
                f"Error - Code: {data.get('code')}, Status: {data.get('status')}, Warnings: {data.get('warnings')}")
            break

        elif isinstance(data.get("data"), list):
            page = data.get("data", [])

            # Write each page to a separate file in the file_path
            os.makedirs(file_path, exist_ok=True)
            temp_file_path = os.path.join(file_path, f"{file_name}_{file_index}.json")
            with open(temp_file_path, "w") as file:
                json.dump(page, file)

            # Check if there is a next page
            if not data.get("has_more", False):
                break

            url = data.get("next_page", "")
            file_index += 1
            time.sleep(0.15)  # Delay to avoid being banned
        else:
            print("This is not a list object, please check the Scryfall queries used.")
            break

    # Merge all files into one final file

    merged_data = []

    for i in range(file_index + 1):
        temp_file_path = os.path.join(file_path, f"{file_name}_{i}.json")
        if os.path.exists(temp_file_path):
            with open(temp_file_path, "r") as file:
                merged_data.extend(json.load(file))
            # Remove the temporary file after reading
            os.remove(temp_file_path)

    final_file_path = os.path.join(file_path, f"{file_name}.json")
    with open(final_file_path, "w") as final_file:
        json.dump(merged_data, final_file)

    end = time.time()
    print(f"{file_name} generated at {file_path} in {end - start:.4f} seconds")