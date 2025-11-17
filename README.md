# Llama-3.2-Docstring-Generator

# Llama 3.2 Docstring Generator (Python + QLoRA)

This project fine-tunes **Llama 3.2 1B Instruct** on the **CodeSearchNet (Python)** dataset to generate **concise one-line Python docstrings** from function bodies.

# Llama 3.2 Docstring Generator (Python + QLoRA)

This project fine-tunes **Llama 3.2 1B Instruct** on the **CodeSearchNet (Python)** dataset to generate **concise one-line Python docstrings** from function bodies.

It’s a small, modern LLM project that uses:

-  `meta-llama/Llama-3.2-1B-Instruct` as the base model  
-  QLoRA (4-bit, parameter-efficient fine-tuning)  
-  CodeSearchNet Python (`code` → `docstring`)  
-  BLEU / ROUGE for evaluation  

---

##  What this model does

Given a Python function, the model generates a **single-line docstring** summarizing what it does.

Example:

```python
def get_vid_from_url(url):
    """Extracts video ID from URL.
    """
    return match1(url, r'youtu\.be/([^?/]+)') or \
        match1(url, r'youtube\.com/embed/([^/?]+)') or \
        match1(url, r'youtube\.com/v/([^/?]+)') or \
        match1(url, r'youtube\.com/watch/([^/?]+)') or \
        parse_query_param(url, 'v') or \
        parse_query_param(parse_query_param(url, 'u'), 'v')
```
Ground truth:
Extracts video ID from URL.

Model prediction:
Extracts video ID from URL. ✅

# Project Structure

notebook.ipynb – Google Colab notebook used for:

loading data from Hugging Face

preprocessing

QLoRA fine-tuning

evaluation and sample generations

inference.py (optional) – small script to load the fine-tuned adapter and generate docstrings.

This README 🙂

# Dataset

Source: Nan-Do/code-search-net-python

We use:

code as the input (full function body)

docstring as the target (we take the first non-empty line)

# Preprocessing

For each example:
Extract the first non-empty line of docstring.

Treat that as a one-line docstring.

Build a simple instruction-style prompt, e.g.:

Write a one-line Python docstring for this function:

{code}

"""
The model is trained to generate the missing line after """.

We also subsample the dataset (≈ 1,000 train examples, plus small val/test sets) to keep training fast and Colab-friendly.

# Model & Training

Base model: meta-llama/Llama-3.2-1B-Instruct

Fine-tuning method: QLoRA (4-bit) with PEFT
 and bitsandbytes

Hardware: Google Colab GPU (T4/L4-class)

Objective: causal language modeling over the full prompt+target text

Key choices:

Sequence length: max_length = 256–512 tokens

Batch size: small (effective batch controlled via gradient accumulation)

Epochs: 1 (small dataset, model converges quickly)

Optimizer: standard AdamW via transformers.Trainer

# Evaluation

I evaluated the model on a small held-out test subset using:

BLEU (via sacrebleu)

ROUGE-1 / ROUGE-2 / ROUGE-L (via rouge)

Sample scores (on ~50 test examples):

BLEU: ~12.4

ROUGE-1: ~0.78

ROUGE-2: ~0.74

ROUGE-L: ~0.78

For short one-line docstrings, ROUGE is more informative than BLEU.
These scores indicate the model usually captures the main action and important tokens, and often matches the reference docstring exactly.

# More Examples

1) Simple case – exact match

CODE:
  def _convert_date_to_dict(field_date):
      """
      Convert native python ``datetime.date`` object  to a format supported by the API
      """
      return {DAY: field_date.day, MONTH: field_date.month, YEAR: field_date.year}

GROUND TRUTH:
  Convert native python ``datetime.date`` object  to a format supported by the API

PREDICTION:
  Convert native python ``datetime.date`` object  to a format supported by the API

2) Slightly weird original docstring

CODE:
  def sina_xml_to_url_list(xml_data):
      """str->list
      Convert XML to URL List.
      From Biligrab.
      """
      ...

GROUND TRUTH:
  str->list

PREDICTION:
  str->list

3) Occasional failure (echoes instruction instead of docstring)

CODE:
  def save_to_file(self, path, filename, **params):
      """
      Saves binary content to a file with name filename. filename should
      include the appropriate file extension, such as .xlsx or .txt, e.g.,
      filename = 'sample.xlsx'.

      Useful for downloading .xlsx files.
      """
      ...

GROUND TRUTH:
  Saves binary content to a file with name filename. filename should

PREDICTION:
  Write a one-line Python docstring for this function:

  def save_to_file(self, path, filename, **params):

Most outputs are correct or near-correct; a small fraction either output nothing or echo part of the instruction.

# How to Use the Model

You can load the fine-tuned adapter and tokenizer directly from Hugging Face:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_id = "Abdul1102/llama32-1b-python-docstrings-qlora"  # <- change this

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

def make_prompt(code: str) -> str:
    return f"""Write a one-line Python docstring for this function:

{code}

\"\"\""""

def generate_docstring(code: str, max_new_tokens: int = 32):
    prompt = make_prompt(code)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip off the prompt
    if full.startswith(prompt):
        continuation = full[len(prompt):]
    else:
        continuation = full
    continuation = continuation.strip()
    # Cut at first newline or triple quote
    for sep in ['"""', '\n']:
        if sep in continuation:
            continuation = continuation.split(sep)[0]
            break
    return continuation.strip().strip('"').strip()
```

# Limitations & Future Work

Sometimes the model:

echoes part of the instruction instead of producing a real docstring,

or produces an empty string for tricky functions.

It only supports Python and only single-line docstrings in this project.

Trained on a relatively small subset of CodeSearchNet for speed.

Possible improvements:

Train on more examples and/or 2–3 epochs.

Add support for multi-line docstrings.

Extend to other languages (Java, JS, etc.).

Add preference tuning so the model prefers more concise / high-signal descriptions.

# Motivation

This project is meant as a modern, small-scale LLM fine-tuning example:

Uses a current model (Llama 3.2 1B Instruct)

Uses QLoRA adapters instead of full fine-tuning

Runs on a single Colab GPU

Produces a practical tool: auto-generated docstrings for Python functions

