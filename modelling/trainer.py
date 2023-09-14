# few shot learning of basic emotions using setfit

from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
from sklearn.utils import shuffle

train_df = pd.read_csv("train_emotions.csv", sep="\t") # manually annotated
test_df = pd.read_csv("test_emotions.csv", sep="\t")

train_df = shuffle(train_df)
test_df = shuffle(test_df)

training_set = Dataset.from_pandas(train_df)
test_set = Dataset.from_pandas(test_df)

print(train_df)
print(test_df)

# load model from hub
model = SetFitModel.from_pretrained("KBLab/sentence-bert-swedish-cased")

# create trainer with default settings
trainer = SetFitTrainer(
    model = model,
    train_dataset = training_set,
    eval_dataset = test_set,
    loss_class = CosineSimilarityLoss
)

trainer.train()
metrics = trainer.evaluate()
model.push_to_hub("gilleti/emotional-classification")
