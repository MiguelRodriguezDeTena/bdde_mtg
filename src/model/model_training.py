import datasets
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from datasets import Dataset, Features, ClassLabel, Value
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import mlflow


def df_to_dataset(df: DataFrame) -> datasets.Dataset:
    """
    Converts a Spark DataFrame to a Hugging Face Dataset for transformer model training.

    Args:
        df (DataFrame): Spark DataFrame to be converted.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object.
    """
    # Bucketize the 'price' column (convert prices >= 3.0 to 1, others to 0) and remove 'oracle_id'
    df = df.withColumn("price", f.when(f.col("price").cast("float") >= 3.0, 1).otherwise(0))\
    .drop('oracle_id')

    df = df.toPandas()

    # Concatenate the target columns into a single string for each row
    target_columns = ['mana_cost','type_line', 'oracle_text', 'bottomright_value','second_mana_cost','second_type_line','second_oracle_text','second_bottomright_value']
    df["gameplay_text"] = df.apply(lambda x: " ".join([str(x[col]) for col in target_columns]), axis=1)

    df = df[["gameplay_text", "price"]]

    # Convert the Pandas DataFrame into a Hugging Face Dataset object
    features = Features({
        "gameplay_text": Value("string"),
        'price': ClassLabel(names=["Below_3", "Above_3"])
    })

    dataset = Dataset.from_pandas(df, features=features)

    return dataset

def train_model(df:DataFrame, config:dict) -> None:

    """
    Trains a DistilBERT model and logs the model and metrics to MLflow.

    Args:
        df (DataFrame): Spark DataFrame containing training data.
        config (dict): Dictionary containing model configuration, tokenizer arguments, and training parameters.

    Returns:
        None
    """

    dataset = df_to_dataset(df)

    # Load the pre-trained DistilBERT model from Hugging Face using the provided config
    distilbert_model = config["distilbert_model"]
    tokenizer = AutoTokenizer.from_pretrained(distilbert_model)

    # Define a tokenization function that uses the tokenizer with specified arguments from config
    def tokenization(x):
        tokenizer_args = config["tokenizer_args"]
        return tokenizer(x["gameplay_text"], **tokenizer_args)

    tokenized_dataset = dataset.map(tokenization, batched= True, remove_columns = ["gameplay_text"])
    tokenized_dataset = tokenized_dataset.rename_column("price", "label").select_columns(["input_ids", "attention_mask", "label"])

    auto_config = AutoConfig.from_pretrained(distilbert_model)

    # Define label mapping (id to label and label to id)
    auto_config.id2label = {0: "Below_3", 1: "Above_3"}
    auto_config.label2id = {"Below_3": 0, "Above_3": 1}

    model = AutoModelForSequenceClassification.from_pretrained(distilbert_model, config=auto_config)

    # Create a data collator to pad sequences to the same length for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Split the dataset into training and testing datasets with config
    ds = tokenized_dataset.train_test_split(config["test_size"])
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Define the training arguments, such as batch size, number of epochs, etc., from the config
    training_args = TrainingArguments(
        **config["training_args"]
    )
    # Initialize MLflow for tracking
    mlflow.set_experiment(f"{config['experiment_path']}/{config['model_name']}")

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model", distilbert_model)
        mlflow.log_param("num_train_epochs", training_args.num_train_epochs)

        # Initialize Hugging Face Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=data_collator
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        evaluation_results = trainer.evaluate()

        # Log evaluation metrics
        mlflow.log_metrics({
            **evaluation_results
        })

        print("Evaluation results:", evaluation_results)

        # Log the trained model to MLflow
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="transformers-model",
            registered_model_name=config["model_name"]
        )

        # Print the run details
        print(f"MLflow Run ID: {run.info.run_id}")