import json
from datetime import datetime, timedelta
import urllib.parse
import requests
import time
import pyspark

def api_call(spark, mode:str, root_dir:str, identifier:str, predict_days:int=30, train_days:int=365) -> None:
    '''
       Make an API call to Scryfall, handle pagination, and save the results to a file.

       Parameters:
       spark (SparkSession): The Spark session.
       mode (str): The mode of operation, either 'train', 'predict' (or 'test')
       root_dir (str): The root directory where the results will be saved.
       identifier (str): A unique identifier for the file name, e.g., a timestamp or ID.
       predict_days (int, optional): Number of days prior to the current date for prediction. Default is 30 days.
       train_days (int, optional): Number of days prior to the current date for training. Default is 365 days.
       '''

    file_path = f"{root_dir}/bronze/{mode}"
    file_name = f"{mode}_bronze_{identifier}"

    # Construct query strings for API requests depending on the mode
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
    temp_data = []

    while True:
        try:
            # Send GET request to the Scryfall API
            response = requests.get(url)
            response.raise_for_status() # Raise exception if the request was unsuccessful
        except requests.RequestException as e:
            print(f"Request failed: {e}") # Print the error message and exit loop if an error occurs
            break

        data = response.json()

        if data.get("object") == "error":
            print(
                f"Error - Code: {data.get('code')}, Status: {data.get('status')}, Warnings: {data.get('warnings')}")
            break

        elif isinstance(data.get("data"), list):
            page = data.get("data", [])
            temp_data.extend(page) # Append the page's data to the temp_data list

            # Check if there is a next page
            if not data.get("has_more", False):
                break

            url = data.get("next_page", "")
            file_index += 1
            time.sleep(0.15)  # Delay to avoid being banned

        else:
            print("This is not a list object, please check the Scryfall queries used.")
            break

    # Store the temporary data as a JSON file in a temporary directory on DBFS (Databricks file system)
    temp_file_path = f"/dbfs/tmp/{file_name}.json"
    with open(temp_file_path, "w") as final_file:
        json.dump(temp_data, final_file)

    #read and write in spark as json
    df = spark.read.format("json").load(f"/tmp/{file_name}.json")
    df.write.format("json").save(f"{file_path}/{file_name}.json", mode="overwrite")

    end = time.time()
    print(f"{file_name} generated at {file_path} in {end - start:.4f} seconds")