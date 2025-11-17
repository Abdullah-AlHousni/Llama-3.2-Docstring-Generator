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

Extracts video ID from URL. 

---

# Project Structure

- **docstring_generator.ipynb** – Google Colab notebook used for:
  - loading data from Hugging Face  
  - preprocessing  
  - QLoRA fine-tuning  
  - evaluation and sample generations  

- **inference.py** – small script to load the fine-tuned adapter and generate docstrings.

- **README.md** – project documentation


## Dataset

Dataset used: **Nan-Do/code-search-net-python** on Hugging Face.

We use:
- `code` as input  
- First non-empty line of `docstring` as target  

Training prompt style (simplified for clarity):

Write a one-line Python docstring for this function:
{function source}

"""

Only the missing line after the triple quotes is learned.

To keep training lightweight, around ~1,000 samples were used.

---

## Model & Training

- Base model: meta-llama/Llama-3.2-1B-Instruct  
- Fine-tuning: QLoRA with 4-bit quantization  
- Frameworks: Transformers, PEFT, Datasets  
- Hardware: Google Colab T4/L4 GPU  
- Sequence length: 256–512 tokens  
- Epochs: 1  
- Objective: causal LM  

QLoRA allows effective fine-tuning on small GPUs by training only low-rank adapter layers while keeping the base model 4-bit quantized.

---

## 📊 Evaluation

Evaluated on a 50-sample test subset using BLEU and ROUGE.

Results:
- BLEU ≈ **12.4**  
- ROUGE-1 ≈ **0.78**  
- ROUGE-2 ≈ **0.74**  
- ROUGE-L ≈ **0.78**  

ROUGE scores above 0.7 indicate strong semantic overlap.  
BLEU is expected to be low for single-line summarization tasks.

---

# More Examples

**Example 1 – Exact match**

CODE:
```
  def _convert_date_to_dict(field_date):
      """
      Convert native python ``datetime.date`` object  to a format supported by the API
      """
      return {DAY: field_date.day, MONTH: field_date.month, YEAR: field_date.year}
```
GROUND TRUTH:

    Convert native python ``datetime.date`` object  to a format supported by the API

PREDICTION:

    Convert native python ``datetime.date`` object  to a format supported by the API

**Example 2 – Exact match**

CODE:
```
  def sina_xml_to_url_list(xml_data):
      """str->list
      Convert XML to URL List.
      From Biligrab.
      """
      ...
```
GROUND TRUTH:

    str->list

PREDICTION:

    str->list

**Example 4 – Occasional failure**

CODE:
```
  def save_to_file(self, path, filename, **params):
      """
      Saves binary content to a file with name filename. filename should
      include the appropriate file extension, such as .xlsx or .txt, e.g.,
      filename = 'sample.xlsx'.

      Useful for downloading .xlsx files.
      """
      ...
```

GROUND TRUTH:

    Saves binary content to a file with name filename. filename should

PREDICTION (Echoed part of the instruction instead of summarizing the code):

    Write a one-line Python docstring for this function:
    def save_to_file(self, path, filename, **params):

Most outputs are correct or near-correct; a small fraction either output nothing or echo part of the instruction.

## How to Use the Model

1. Load the base model and tokenizer  
2. Load the fine-tuned QLoRA adapter  
3. Create a prompt:  
   Write a one-line Python docstring for this function:  
   {code}  
   """  
4. Generate continuation and clean up quotes/newlines  

You can load the fine-tuned adapter and tokenizer directly from Hugging Face:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_id = "Abdul1102/llama32-1b-python-docstrings-qlora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

def make_prompt(code: str) -> str:
    return f'Write a one-line Python docstring for this function:\n\n{code}\n\n"""'

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

## ⚠️ Limitations

- Occasionally outputs empty strings or echoes the prompt  
- Limited to Python  
- Supports only one-line docstrings  
- Dataset docstrings are noisy, which can affect BLEU scores  

---

## Future Work

- Multi-epoch training for more consistency  
- Multi-line docstring generation  
- Larger model (3B) for robustness  
- Add a Gradio demo  
- Extend to Java, JavaScript, or other languages  

---

## Motivation

This project demonstrates:
- Modern low-resource fine-tuning (QLoRA + 4-bit)  
- Training a real, practical model from open data  
- End-to-end pipeline: preprocessing → fine-tuning → evaluation → HF upload  
- A clean, reproducible workflow for LLM fine-tuning projects  
