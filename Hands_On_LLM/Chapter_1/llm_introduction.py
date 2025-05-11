
# Loading model and tokenizer

# Phi-3 

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct" , # model name
    device_map="cuda" , # device where the model will be run gpu(cuda) or cpu
    torch_dtype="auto", # datatype of model's parameters
    trust_remote_code=False, # run models which are only available on hub 
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


# instead of using model and tokenizer separately , you can use them using single command

from transformers import pipeline

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # prompt is not returned , only gnerated output
    max_new_tokens=500, # max number of generated tokens
    do_sample=False # If false, the model returns only the most pobably token
)

# The prompt (user input / query)

messages = [ 
    {"role": "user", "content": " Create a funny joke about chickens."}
]

# Generate output
output = generator(messages)
print(output[0]["generated_text"])