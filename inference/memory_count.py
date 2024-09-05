from pynvml import *
import transformers


def print_gpu_utilization():
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("Device {}: {}".format(i, nvmlDeviceGetName(handle)))
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")

print("Start of the program")
print_gpu_utilization()

llama_path = "/work/10159/rmarten/ls6/dcft/llama-7b"
print(f"Loading {llama_path}")

model = transformers.AutoModelForCausalLM.from_pretrained(llama_path)
model.to('cuda')

print(f'Model is on device {model.device}')
print("After loading model")
print_gpu_utilization()


tokenizer = transformers.AutoTokenizer.from_pretrained(llama_path)
print("After loading tokenzier")
print_gpu_utilization()

# Test the model
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
input_ids = input_ids.to('cuda')
print(f'Inputs are on device {input_ids.device}')
print_gpu_utilization()

# Generate a response
output = model.generate(input_ids, max_length=50)
print(f'Outputs are on device {output.device}')

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Output: {generated_text}")

