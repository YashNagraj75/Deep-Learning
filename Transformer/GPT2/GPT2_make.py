from dataclasses import dataclass
import torch,math,tiktoken,time,inspect
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Key, Query, Value projections for all heads.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) 
        # This is the output projection for all heads.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # This is the look ahead mask but given the name of bias
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) # This one return the lower triangle matrix given the tensor
                             .view(1, 1, config.block_size, config.block_size)) # This reconstructs the 2D tensor into a 4D tensor
        
    def forward(self,x): # The forward pass
        B,T,C  = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        # So each head * head_size 

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2) 
        # Here the C // self.n_heads splits the Channels into n_heads Ex: 768 / 12 = 64
        # So the dim becomes (B,nh,T,nh)

        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2) 
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2) 

        # Now plugin the attention equation: (q * kT)/(sqrt(dk)). We will next multiply the values
        # attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) 
        # attn = F.softmax(attn, dim=-1)
        # # Okay so here when masked_fill is applied we set the corresponding values of the sequence to 0 and then setting these zeros to -inf
        # # This is done in order to avoid the model to look at these weights and then decrease learning
        # y = attn @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y
        

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPT2Config): 
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing 
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            """Here we scale the residual layer weights by a factor of -0.5 following the GPT2 paper
            Here we multiply the num_layers by 2 because we have 2 residual connections
            1) From the MHA blocks
            2) From the FFN blocks

            We do this using a flag NANOGPT_SCALE_INIT flag
            """
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits , loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # print([i for i in sd_keys_hf + sd_keys if i not in sd_keys_hf or i not in sd_keys])
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self,weight_decay,learning_rate, device):
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad} # Gets only the params that need gardient

        # Just for clarification weight decay is nothing but L2 norm for regularization purposes
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _,p in param_dict.items() if p.dim() < 2] # These params are the ones like biases and layer norms that don't require to be decayed 
        optim_grops = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"Num decay params {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non_decay_prams {len(nodecay_params)}, with {num_nodecay_params} parameters")
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused and 'cuda' in device
        print(f"Using fused: {used_fused}")
        optimizer = torch.optim.AdamW(optim_grops, lr=learning_rate,betas=(0.9,0.95), eps=1e-8, fused=True)

        return optimizer

# ----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self,B, T):
        self.B = B
        self.T = T

        with open('input.txt','r') as f:
            text = f.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"Per epoch we have {len(self.tokens) // (self.B * self.T)} batches")

        # Initial position 
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = buf[:-1].view(B,T).to("cuda")
        y = buf[1:].view(B,T).to('cuda')

        self.current_position += B*T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x ,y 


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')
model = GPT(GPT2Config(vocab_size=50304))
model.eval()
model.to(device)

num_repeat_sequences = 5
max_length = 30
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 1024 #524288
B = 4 # 16
T = 32 # 1024 sequence length
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B = 4, T = 32)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it): # Here we are using the linear warmup and then cosine decay
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0<=decay_ratio<=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Optimization
optimizer = model.configure_optimizer(weight_decay=0.1,learning_rate=6e-4,device=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0  
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits , loss= model(x,y)
            
        loss.backward()
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    optimizer.step()
    torch.cuda.synchronize()

    # Clipping gradients to prevent the exploding gradient problem
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Determining the learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0 ) * 1000    
    print(f"step {step:4d} |  loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.6f} | dt {dt:.2f}ms")

import sys;sys.exit() 



# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# epsilon = 1e-8
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x)
#         logits = logits[:, -1, :] 

#         probs = F.softmax(logits, dim=-1)

#         topk_probs, topk_indices = torch.topk(probs, 20,dim=-1)


#         ix = torch.multinomial(topk_probs,1)

#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)

# # Print the generated text
# for i in range(num_repeat_sequences):
#     print(">",tokenizer.decode(x[i].tolist()))
