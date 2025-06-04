import sys
sys.path.append('../src')
from src.lora_model import LORAEngineGeneration
from src.influence import IFEngineGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import warnings
warnings.filterwarnings("ignore")

project_path = "/home/vramineni/DataInf"

from datasets import load_dataset, DatasetDict, Dataset

# Step 1: Load the dataset
dataset = load_dataset("suolyer/pile_youtubesubtitles")

data_name =  'pile_youtubesubtitles'

dataset = load_dataset(f"suolyer/{data_name}")
#dataset = load_dataset(f"Self-GRIT/{data_name}")


# Step 2: Save the entire dataset (DatasetDict)
#dataset.save_to_disk("/home/vramineni/DataInf/datasets/pile_youtubesubtitles.hf")
dataset.save_to_disk(f"/home/vramineni/DataInf/datasets/{data_name}.hf")


# Step 3: Load the dataset from disk (DatasetDict)
dataset_dict = DatasetDict.load_from_disk(f"/home/vramineni/DataInf/datasets/{data_name}.hf")

# Step 4: Access individual splits
validation_dataset = dataset_dict['validation']
test_dataset = dataset_dict['test']

# Optional: Save only the train split if you only need that
validation_dataset.save_to_disk(f"/home/vramineni/DataInf/datasets/{data_name}_train.hf")
test_dataset.save_to_disk(f"/home/vramineni/DataInf/datasets/{data_name}_test.hf")



# You can specify the Qwen model from Hugging Face by changing the model initialization

# Load the Qwen model and tokenizer from Hugging Face

model_name = "allenai/open-instruct-pythia-6.9b-tulu"
#model_name = "EleutherAI/pythia-14m" # Update with the Qwen model identifier from Hugging Face
#model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Now initialize the LORAEngineGeneration with the new base_path and model settings
lora_engine = LORAEngineGeneration(base_path=model_name, 
                                   project_path=project_path,
                                   dataset_name=data_name)


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

prompt = """
Who would make a better leader, a man or a woman?
"""
inputs = lora_engine.tokenizer(prompt, return_tensors="pt")


# Generate
generate_ids = lora_engine.model.generate(input_ids=inputs.input_ids, 
                                          max_length=128,
                                          pad_token_id=lora_engine.tokenizer.eos_token_id)
output = lora_engine.tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

print('-'*50)
print('Print Input prompt')
print(prompt)
print('-'*50)
print('Print Model output')
print(output)
print('-'*50)

tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)

influence_engine = IFEngineGeneration()
influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
influence_engine.compute_hvps()
influence_engine.compute_IF()

most_influential_data_point_proposed=influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmax(), axis=1)
least_influential_data_point_proposed=influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmin(), axis=1)

val_id=21
max_length = 300
print(f'Validation Sample ID: {val_id}\n', 
      lora_engine.validation_dataset[val_id]['text'][:max_length], '\n')
print('The most influential training sample: \n', 
      lora_engine.train_dataset[int(most_influential_data_point_proposed.iloc[val_id])]['text'][:max_length], '\n')
print('The least influential training sample: \n', 
      lora_engine.train_dataset[int(least_influential_data_point_proposed.iloc[val_id])]['text'][:max_length])


