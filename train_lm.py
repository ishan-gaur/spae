import torch
from tinylm import TinyLM
from datasets import load_dataset
from data import DocumentStream
from torchinfo import summary

lm = TinyLM()
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)
def encode(examples):
    tokens = []
    for e in examples:
        sample_tokens = lm.tokenize(e)
        # tokens.append(next(iter(lm.tokenizer._special_tokens.values())))
        sample_tokens.append(lm.tokenizer._special_tokens["<|endoftext|>"]) # FIXME GPT2 tokenizer specific!
        tokens.append(sample_tokens)
    return {"input_ids": tokens}

print("Splitting Dataset")
dataset_splits = dataset.train_test_split(test_size=0.0001)

print("Data split lengths")
print(len(dataset_splits["train"]), len(dataset_splits["test"]))


print("Tokenizing Dataset")
# tokenized_train_data = dataset_splits["train"].map(
#     encode, batched=True, input_columns="text", 
#     remove_columns=["id", "dump", "url", "file_path", "language", "language_score", "token_count", "score", "int_score"]
# )
tokenized_train_data = dataset_splits["test"].map(
    encode, batched=True, input_columns="text", 
    remove_columns=["id", "dump", "url", "file_path", "language", "language_score", "token_count", "score", "int_score"],
    num_proc=12
)

train_data_stream = DocumentStream(tokenized_train_data, sample_len=4096, epochs=1)
data_iter = iter(train_data_stream)
print(next(data_iter))
print(len(next(data_iter)))

ct = 0
toks = 0
for s in train_data_stream:
    ct += 1
    toks += len(s)    

print(f"Total tokens {toks} and total samples {ct}, total in dataset {len(tokenized_train_data)}, mean token count {sum(dataset_splits["test"]["token_count"]) / len(dataset_splits["test"])}")
# summary(lm, input_data=torch.tensor([lm.tokenize(train_data[0]["text"])[:4096]]).int())
