from llama_cpp import Llama

# Replace with the actual path to your GGUF model
model_path = "C:\Users\admin\chat\models\phi-2.Q4_K_M.gguf"
llm = Llama(model_path=model_path)

output = llm("Hello!", max_tokens=50)
print(output)
