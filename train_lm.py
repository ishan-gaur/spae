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
# dataset_splits = dataset.train_test_split(test_size=0.0001)
dataset_splits = dataset.train_test_split(test_size=0.999999)

print("Data split lengths")
print(len(dataset_splits["train"]), len(dataset_splits["test"]))


print("Tokenizing Dataset")
tokenized_train_data = dataset_splits["train"].map(
    encode, batched=True, input_columns="text", 
    remove_columns=["id", "dump", "url", "file_path", "language", "language_score", "token_count", "score", "int_score"],
    num_proc=12
)

context_len = 32
train_data_stream = DocumentStream(tokenized_train_data, sample_len=context_len + 1, epochs=1)
opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)

torch.autograd.set_detect_anomaly(True)

s = 0
for sample in train_data_stream:
    s += 1
    input_toks = sample[:-1]
    output_toks = sample[1:]
    logits = lm(input_toks[None, ...]).squeeze() # lm assumes a batch dimension
    loss = torch.nn.functional.cross_entropy(logits, output_toks)
    print(f"Sample: {s}\tLoss: {loss}")
    opt.zero_grad()
    loss.backward()
    opt.step()
