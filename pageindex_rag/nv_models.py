from client import NVIDIA_API_KEY
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-30b-a3b",
    nvidia_api_key=NVIDIA_API_KEY,
    temperature=0.0,
    max_completion_tokens=1024,
)

llm = ChatNVIDIA()
models = llm.available_models
print(model.id for model in models) 

with open("models.txt", "w") as f:
    f.write("\n".join(str(model.id) for model in models)) 

