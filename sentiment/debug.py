from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer


import numpy as np
from datasets import load_metric, load_dataset, Dataset

import pandas as pd
import os

import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

df = pd.read_csv("testfile.txt", delimiter=",", quotechar="|")

df.loc[df['label'] < 0, 'label'] = 2

tweets = Dataset.from_pandas(df)
tweets = tweets.train_test_split(test_size=0.1)

model = AutoModelForSequenceClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=3)


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}
    
    
    small_train_dataset = tweets["train"].shuffle(seed=42).select([i for i in list(range(2000))])
small_test_dataset = tweets["test"].shuffle(seed=42).select([i for i in list(range(200))])


tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased", model_max_length=512)


tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)


def preprocess_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
    
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)


repo_name = "finetuning-sentiment-model-test"


trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_test)


training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=4,
   per_device_eval_batch_size=4,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=False,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)


trainer.train()
