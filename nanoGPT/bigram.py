import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # How many independent sequences will we process in parrallel (Channels - C)
block_size = 8 # What is the maximum context lenght for predicitons (Token - T)
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.2
#-----------------

torch.manual_seed(1337)

# Read in our dataset
with open('/Users/windu/Developer/MachineLearning/nanoGPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all the unique characters from the dataset and put it in a list
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers (Encode/Decode Pairs)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Create our training and test data splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n] # First 90%
val_data = data[n:] # Last 10%

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random offets of length 4 inside the training data to grab our batches
    x  = torch.stack([data[i:i+block_size] for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

# Estimate our loss over multiple batches for both splits (not affected by .backward() calls)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Single Head Attention
class Head(nn.Module):
    """ One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)    # (B,T,C)
        q = self.query(x)  # (B,T,C)
        
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #(B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T ,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) ----> (B, T, C)
        return out

# Multihead Attention
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel that concat their results together"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    """allows the tokens to think about itsef (computation)"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # feed foward linear layers have 4x the outside linear layers
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: comunicaiton followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # Communicaiton
        self.ffwd = FeedForward(n_embd) # Computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Communicaiton with residual connection ( the + ) & layernorm before self attention
        x = x + self.ffwd(self.ln2(x)) # Computation with residual conneciton ( the + ) & layernorm before feed forward
        return x

# Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # we want to embedd the indexes of our tokens
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets = None):
        B, T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) tensor with C being our logits (B = Batch, T = Time, C = Channel) (4,8,65) 65 is our vocab size
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C) x now holds not only token identities, but the positions in which they occur
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B,T, C = vocab_size)

        # if no targets, there is no loss to generate
        if targets is None:
            loss = None
        else:
            #manipluate or (B,T,C) tensor into a (BT, C) tensor for pytorch cross_entropy compatability
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # now two dimensional 
            targets = targets.view (B*T) # now one dimensional
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the curent context
        for _ in range(max_new_tokens):
            # grop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution of probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # ( B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch Optimizter
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evalulate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate tokens from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))