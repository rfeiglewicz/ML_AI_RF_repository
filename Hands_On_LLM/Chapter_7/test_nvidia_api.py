import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
result = llm.invoke("Write a ballad about LangChain.")
print(result.content)