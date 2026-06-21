# Deep Learning

A personal collection of deep learning experiments, implementations, and explorations covering word embeddings, recurrent sequence models, attention mechanisms, transformer architectures from scratch, and LLM fine-tuning. Projects span both TensorFlow/Keras and PyTorch.

---

## Topics Covered

- Word embeddings (GloVe), cosine similarity, analogy generation, and gender-bias debiasing
- Sequence classification with LSTMs using pre-trained embeddings (Emojify)
- Attention-based Neural Machine Translation (NMT) with Bidirectional LSTMs
- Transformer architecture built from scratch (encoder, decoder, positional encodings, multi-head attention)
- GPT-2 replica trained from scratch with Distributed Data Parallel (DDP) and Flash Attention
- Full fine-tuning and PEFT/LoRA fine-tuning of Flan-T5 on dialogue summarization
- Question answering using DistilBERT on the bAbI dataset
- NVIDIA inference API experimentation
- PyTorch fundamentals (tensors, modules, optimizers, training loops)

---

## Tech Stack

| Library | Usage |
|---|---|
| TensorFlow / Keras | NMT, Emojify, Transformer architecture |
| PyTorch | GPT-2 from scratch, GAN sandbox, tensor experiments |
| Hugging Face Transformers | Flan-T5 fine-tuning, DistilBERT QA |
| Hugging Face PEFT | LoRA adapter fine-tuning |
| Hugging Face Datasets | DialogSum, bAbI QA, FineWeb-Edu |
| NumPy | GloVe embedding operations, data preprocessing |
| tiktoken | GPT-2 tokenization |
| NVIDIA NIM API (OpenAI-compatible) | Inference with Nemotron-4 340B |

---

## Project Structure

```
Deep-Learning/
├── Embeddings/
│   ├── Debiasing.ipynb        # GloVe word vector exploration, cosine similarity, word analogies,
│   │                          # and gender-bias neutralization/equalization of embeddings
│   └── Emojify.ipynb          # Two emoji classifiers:
│                              #   v1 - softmax over averaged GloVe vectors (numpy only)
│                              #   v2 - LSTM with pre-trained GloVe embedding layer (Keras/TF)
│
├── NMT/
│   └── neural-mt.ipynb        # Sequence-to-sequence date normalization model
│                              # Bidirectional LSTM encoder + attention mechanism +
│                              # LSTM decoder. Converts human-readable dates (e.g.,
│                              # "Saturday May 9 2018") to ISO format ("2018-05-09").
│                              # Trained on 10,000 synthetically generated examples.
│
├── Transformer/
│   ├── architecture/
│   │   ├── transformer.ipynb  # Full encoder-decoder Transformer built from scratch in TF/Keras.
│   │   │                      # Implements positional encodings, scaled dot-product attention,
│   │   │                      # multi-head attention, padding masks, and look-ahead masks.
│   │   └── qagen-using-transformers.ipynb
│   │                          # Question-answering system using DistilBERT (TF) fine-tuned
│   │                          # on the bAbI QA dataset. Covers dataset processing, tokenizer
│   │                          # alignment, and span extraction training.
│   │
│   ├── GPT2/
│   │   ├── GPT2_make.py       # GPT-2 (124M) replicated from scratch in PyTorch.
│   │   │                      # Includes: causal self-attention with Flash Attention,
│   │   │                      # weight-tied token/position embeddings, cosine LR schedule
│   │   │                      # with linear warmup, gradient clipping, bfloat16 training,
│   │   │                      # gradient accumulation, and full DDP multi-GPU support.
│   │   │                      # Supports loading pretrained HuggingFace GPT-2 weights.
│   │   ├── fineweb.py         # Data pipeline: downloads and tokenizes the FineWeb-Edu
│   │   │                      # 10B-token dataset into shards of 100M tokens using
│   │   │                      # multiprocessing and tiktoken.
│   │   ├── test.ipynb         # Notebook for interactive GPT-2 experiments
│   │   ├── input.txt          # Sample text corpus for local training runs
│   │   └── flake.nix          # Nix flake for reproducible dependency management
│   │
│   ├── fine-tuning.ipynb      # Full fine-tune and PEFT/LoRA fine-tune of google/flan-t5-base
│   │                          # on the DialogSum conversation summarization dataset.
│   │                          # Uses HuggingFace Trainer. LoRA targets q and v projection
│   │                          # matrices (r=32, alpha=32).
│   ├── fine_tuning.py         # Script version of the same Flan-T5 fine-tuning workflow
│   └── flan-t5-lora/          # Saved LoRA adapter checkpoint (PEFT 0.11.1)
│                              # Base model: google/flan-t5-base
│
├── GAN/
│   └── torch_test.ipynb       # PyTorch fundamentals sandbox: tensor operations, shapes,
│                              # nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Sequential,
│                              # custom nn.Module subclassing, Adam optimizer, training loop.
│
├── data/
│   ├── train_emoji.csv        # 132 labeled training sentences for emoji classification
│   ├── test_emoji.csv         # 56 labeled test sentences for emoji classification
│   └── tesss.csv              # Additional test split
│
├── models/                    # Saved Keras model files (tracked via Git LFS)
│   ├── Emojify_v2.keras
│   └── Neural_Machine_Translation.keras
│
└── nvidia-nmi.ipynb           # Calls NVIDIA NIM API (OpenAI-compatible endpoint) to run
                               # inference with nvidia/nemotron-4-340b-instruct
```

---

## Installation and Setup

Most notebooks were developed on Kaggle or a local GPU machine. To run locally:

```bash
# Clone the repository
git clone https://github.com/YashNagraj75/Deep-Learning.git
cd Deep-Learning
```

### Python dependencies

```bash
pip install torch torchvision torchaudio          # PyTorch
pip install tensorflow                            # TensorFlow/Keras
pip install transformers datasets peft evaluate   # HuggingFace ecosystem
pip install tiktoken tqdm numpy matplotlib faker babel emoji
pip install openai python-dotenv                  # NVIDIA NIM API notebook
```

For the GPT-2 project, a Nix flake is provided (`Transformer/GPT2/flake.nix`) to pin exact dependency versions.

### Data requirements

- **GloVe embeddings**: `glove.6B.50d.txt` (download from [nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)). Notebooks expect it at `data/glove.6B.50d.txt` or an absolute path.
- **Emoji data**: `data/train_emoji.csv` and `data/tesss.csv` are included in the repository.
- **FineWeb-Edu**: Downloaded automatically by `fineweb.py` from HuggingFace.
- **DialogSum, bAbI**: Downloaded automatically via `datasets.load_dataset(...)`.
- **DistilBERT tokenizer**: The QA notebook references a local tokenizer path (`/kaggle/input/tokeniser`); replace with `distilbert-base-uncased` for local use.

---

## Usage

### Run a notebook

Open any `.ipynb` file in Jupyter or VS Code. Each notebook is self-contained and documents its own flow.

```bash
jupyter notebook Embeddings/Emojify.ipynb
```

### Train GPT-2 from scratch (single GPU)

```bash
python Transformer/GPT2/GPT2_make.py
```

### Train GPT-2 with Distributed Data Parallel (multi-GPU)

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> Transformer/GPT2/GPT2_make.py
```

### Fine-tune Flan-T5 with LoRA

```bash
python Transformer/fine_tuning.py
```

### Prepare FineWeb-Edu dataset shards

```bash
python Transformer/GPT2/fineweb.py
```

---

## Notes

- The GPT-2 implementation supports loading pretrained HuggingFace weights via `GPT.from_pretrained('gpt2')`.
- The LoRA adapter checkpoint in `Transformer/flan-t5-lora/` can be loaded with `peft.PeftModel` on top of `google/flan-t5-base`.
- Large binary files (`.keras` model weights) are tracked with Git LFS.
- Several notebooks contain hardcoded absolute paths from the original development environment; update these to match your local setup before running.
