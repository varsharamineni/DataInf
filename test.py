import sys
sys.path.append('../src')
from src.lora_model import LORAEngineGeneration
from src.influence import IFEngineGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


import warnings
warnings.filterwarnings("ignore")


project_path = "/leonardo/home/userexternal/vraminen/DataInf"

# You can specify the Qwen model from Hugging Face by changing the model initialization

# Load the Qwen model and tokenizer from Hugging Face
model_name = "Qwen/Qwen2.5-Math-1.5B"  # Update with the Qwen model identifier from Hugging Face
#model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Now initialize the LORAEngineGeneration with the new base_path and model settings
lora_engine = LORAEngineGeneration(base_path=model_name, 
                                   project_path=project_path,
                                   dataset_name='math_without_reason')


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

prompt = """
Emily scored 10 points in the first game, 30 points in the second, 100 in the third, and 20 in the fourth game. What is her total points? Output only the answer.
"""
inputs = lora_engine.tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
generate_ids = lora_engine.model.generate(input_ids=inputs.input_ids, 
                                          max_length=128,
                                          pad_token_id=lora_engine.tokenizer.eos_token_id)
output = lora_engine.tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]

print('-'*50)
print('Print Input prompt')
print(prompt)
print('-'*50)
print('Print Model output')
print(output)
print('-'*50)