import torch
from tinylm import TinyLM
from datasets import load_dataset
from torchinfo import summary

lm = TinyLM()
train_data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)
print(train_data)

summary(lm, input_data=torch.tensor([lm.tokenize(train_data[0]["text"])[:5]]).int())
