# %%

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# %%
# Load dataset

dataset = load_dataset("roneneldan/TinyStoriesInstruct")

# %%
# Load pretrained model for experimentation

model_name = 'roneneldan/TinyStories-Instruct-33M'

tokenizer = AutoTokenizer.from_pretrained(model_name)
pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)

# %%
# Experiment with the pretrained model

# Generate completions
prompt = "The cat sat on the mat"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
greedy_output = pretrained_model.generate(input_ids, max_length=200)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# %%
# Define a transformer to run on the dataset

device = 'cpu' # FIXME: Memory leak on mps

class Transformer(nn.Module):
    def __init__(self, n_embd=512, vocab_size=50257):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        transformer_layer = nn.TransformerEncoderLayer(512, 8)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=8, mask_check=True)
        self.unembed = nn.Linear(n_embd, vocab_size)


    def forward(self, x):
        x = self.embed(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.unembed(x)
        return x 

model = Transformer()
model(input_ids)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%

train_loader = DataLoader(dataset['train'], batch_size=128, shuffle=True)

# %%

tokenizer.pad_token = tokenizer.eos_token

for epoch in range(10):
    for batch in tqdm(train_loader):
        optim.zero_grad()

        tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt')['input_ids'].to(device)
        logits = model(tokenized)
        # flatten out seq dim
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tokenized.view(-1))
        tqdm.write(f"Loss: {loss.item()}")
        loss.backward()
        optim.step()


# %%
