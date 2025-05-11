
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)


# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)


# Inputs and outputs of a trained LLM

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

output = generator(prompt)

print(output[0]['generated_text']) # first batch output

# LLM architectrue

print(model)

prompt = "The capital of France is"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Tokenize the input prompt
input_ids = input_ids.to("cuda")

# Get the output of the model before the lm_head
model_output = model.model(input_ids)

# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])

# token_id = lm_head_output[0,-1].argmax(-1)
# Show top k results
token_id = torch.topk(lm_head_output[0,-1], 10).indices
print(tokenizer.decode(token_id))

print(model_output[0].shape)

print(lm_head_output.shape)


# Speeding up generation by caching keys and values

prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

import time
start_time = time.time()

# Generate the text
generation_output = model.generate(
  input_ids=input_ids,
  max_new_tokens=100,
  use_cache=True
)

print("--- %s seconds -with cache ---" % (time.time() - start_time))

start_time = time.time()

# Generate the text
generation_output = model.generate(
  input_ids=input_ids,
  max_new_tokens=100,
  use_cache=False
)


print("--- %s seconds -without cache ---" % (time.time() - start_time))