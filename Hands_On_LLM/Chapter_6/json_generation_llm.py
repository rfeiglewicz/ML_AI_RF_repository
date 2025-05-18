import gc
import torch

# Flush memory
gc.collect()
torch.cuda.empty_cache()


from llama_cpp.llama import Llama

# Load Phi-3
llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*q4.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False
)

# Generate output
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Create a warrior for an RPG in JSON format."},
    ],
    response_format={"type": "json_object"},
    temperature=0,
)['choices'][0]['message']["content"]

llm.close()
llm._sampler.close()

import json

# Format as json
json_output = json.dumps(json.loads(output), indent=4)
print(json_output)