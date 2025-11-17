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