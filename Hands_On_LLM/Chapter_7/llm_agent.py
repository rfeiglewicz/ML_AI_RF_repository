# Loading an LLM
# wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
# Alternatively use smaller model witch q4 quantization
# wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf

from langchain import LlamaCpp

# Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="Phi-3-mini-4k-instruct-fp16.gguf",
#     n_gpu_layers=-1,
#     max_tokens=500,
#     n_ctx=2048,
#     seed=42,
#     verbose=False
# )

# Q4 quantization
llm = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1, # all layers are deployed on a GPU
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

print(llm.invoke("Hi! My name is Maarten. What is 1 + 1?"))

# Creating simple chain

from langchain import PromptTemplate

# Create a prompt template with the "imput_prompt" variable

template = """<s><|user|>
{input_prompt}<|end|>
<|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"]
)

basic_chain = prompt | llm

# Use the chain
output = basic_chain.invoke(
    {
        "input_prompt": "Hi! My name is Maarten. What is 1 + 1?",
    }
)

print(output)


# Creating multiple chains

from langchain import LLMChain

# Create a chain for the title of our story
template = """<s><|user|>
Create a title for a story about {summary}. Only return the title.<|end|>
<|assistant|>"""
title_prompt = PromptTemplate(template=template, input_variable=["summary"])
title = LLMChain(llm=llm, prompt=title_prompt, output_key="title") # The chain is outputting a dictionary with a key "title"

output =title.invoke({"summary": "a girl that lost her mother"})

print(output)

# Create a chain for the character description using the summary and title
template = """<s><|user|>
Describe the main character of a story about {summary} with the title {title}. Use only two sentences.<|end|>
<|assistant|>"""
character_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title"]
)
character = LLMChain(llm=llm, prompt=character_prompt, output_key="character")

# Create a chain for the story using the summary, title, and character description
template = """<s><|user|>
Create a story about {summary} with the title {title}. The main charachter is: {character}. Only return the story and it cannot be longer than one paragraph<|end|>
<|assistant|>"""
story_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title", "character"]
)
story = LLMChain(llm=llm, prompt=story_prompt, output_key="story")

# Combine all three components to create the full chain
llm_chain = title | character | story

result = llm_chain.invoke("a girl that lost her mother")

print(result)

# Memory
# Let's give the LLM our name
print(basic_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"}))

# Next, we ask the LLM to reproduce the name
print(basic_chain.invoke({"input_prompt": "What is my name?"}))
# LLM does not remember the name because previous conversations are not taking into account


# ConversationBuffer

# Create an updated prompt template to include a chat history
template = """<s><|user|>Current conversation:{chat_history}

{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)

from langchain.memory import ConversationBufferMemory

# Define the type of Memory we will use
memory = ConversationBufferMemory(memory_key="chat_history")

# Chain the LLM, Prompt, and Memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

# Generate a conversation and ask a basic question
print(llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"}))

# Does the LLM remember the name we gave it?
print(llm_chain.invoke({"input_prompt": "What is my name?"}))
# Yes, the llm is aware about previous conversations

# ConversationBufferMemoryWindow

from langchain.memory import ConversationBufferWindowMemory

# Retain only the last 2 conversations in memory
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")

# Chain the LLM, Prompt, and Memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)
print("---------------- Testing Conversation Buffer Window Memory -----------")
# Ask two questions and generate two conversations in its memory
llm_chain.invoke({"input_prompt":"Hi! My name is Maarten and I am 33 years old. What is 1 + 1?"})
llm_chain.invoke({"input_prompt":"What is 3 + 3?"})

# Check whether it knows the name we gave it
print(llm_chain.invoke({"input_prompt":"What is my name?"}))

# Check whether it knows the age we gave it
# Now is not remember my age because the history buffer holding only 2 recent prompts
print(llm_chain.invoke({"input_prompt":"What is my age?"}))

# ConversationSummary
# Instead of pasting all your converastion , use an llm to summarize previous conversations

# Create a summary prompt template
summary_prompt_template = """<s><|user|>Summarize the conversations and update with the new lines.

Current summary:
{summary}

new lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>"""
summary_prompt = PromptTemplate(
    input_variables=["new_lines", "summary"],
    template=summary_prompt_template
)

from langchain.memory import ConversationSummaryMemory

# Define the type of memory we will use
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    prompt=summary_prompt
)

# Chain the LLM, prompt, and memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

print("---------------- Testing Conversation Summary -----------")

# Generate a conversation and ask for the name
print(llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"}))
print(llm_chain.invoke({"input_prompt": "What is my name?"}))

# Check whether it has summarized everything thus far
print(llm_chain.invoke({"input_prompt": "What was the first question I asked?"}))

# Check what the summary is thus far
print(memory.load_memory_variables({}))

# Agents using Nvidia NIM

import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Add your NVIDIA_API_KEY as environment variable
# Load OpenAI's LLMs with LangChain
llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

# Create the ReAct template
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

from langchain.agents import load_tools, Tool
from langchain.tools import DuckDuckGoSearchResults

# You can create the tool to pass to an agent
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duckduck",
    description="A web search engine. Use this to as a search engine for general queries.",
    func=search.run,
)

# Prepare tools
tools = load_tools(["llm-math"], llm=llm)
tools.append(search_tool)

from langchain.agents import AgentExecutor, create_react_agent

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# What is the Price of a MacBook Pro?
print(agent_executor.invoke(
    {
        "input": "What is the current price of a MacBook Pro in USD? How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
    }
))