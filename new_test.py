import sys
sys.path.append('../src')
from src.lora_model import LORAEngineGeneration
from src.influence import IFEngineGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")


project_path = "/home/vramineni/DataInf"

# Load the new dataset from Hugging Face's datasets library
dataset_name = "suolyer/pile_youtubesubtitles"
dataset = load_dataset(dataset_name)

# Load the model and tokenizer from Hugging Face
model_name = "EleutherAI/pythia-14m"  # Update with the model identifier from Hugging Face
# You can load the model and tokenizer using AutoModelForCausalLM and AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Now initialize the LORAEngineGeneration with the new base_path and model settings
lora_engine = LORAEngineGeneration(base_path=model_name,
                                   project_path=project_path,
                                   dataset_name=dataset_name)  # Update dataset_name

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Update this to match the format of the new dataset
prompt = """
Emily scored 10 points in the first game, 30 points in the second, 100 in the third, and 20 in the fourth game. What is her total points? Output only the answer.
"""
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response from the model
generate_ids = model.generate(input_ids=inputs.input_ids, 
                              max_length=128,
                              pad_token_id=tokenizer.eos_token_id)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

# Print out the results
print('-'*50)
print('Print Input prompt')
print(prompt)
print('-'*50)
print('Print Model output')
print(output)
print('-'*50)

# Process the new dataset for tokenization
# Make sure to customize the tokenization method according to the dataset's structure
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Prepare collate_fn for padding the dataset during training or inference
collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")

# Now, compute the gradients with the LORAEngineGeneration
tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)

# Influence computation
influence_engine = IFEngineGeneration()
influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
influence_engine.compute_hvps()
influence_engine.compute_IF()

# Get most and least influential data points
most_influential_data_point_proposed = influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmax(), axis=1)
least_influential_data_point_proposed = influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmin(), axis=1)

# Example of printing out influential data points
val_id = 21
print(f'Validation Sample ID: {val_id}\n', 
      lora_engine.validation_dataset[val_id]['text'], '\n')
print('The most influential training sample: \n', 
      lora_engine.train_dataset[int(most_influential_data_point_proposed.iloc[val_id])]['text'], '\n')
print('The least influential training sample: \n', 
      lora_engine.train_dataset[int(least_influential_data_point_proposed.iloc[val_id])]['text'])
