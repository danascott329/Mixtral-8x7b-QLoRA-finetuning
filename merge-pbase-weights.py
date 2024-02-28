from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import pandas as pd
import os

cache_dir = "/vast/palmer/scratch/odea/das293/huggingface/"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir


device = "auto"
weights = "fulltext-lora-weights/model_weights"             # Path to the LoRA weights exported from Predibase
base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"        # fine-tuned linewise model

# Prompt should be in this style due to how the data was created
#prompt = "#### Human: What is the capital of Australia?#### Assistant:"

prompt = f'''
Your task is to summarize the company’s telework policy in a way that is understandable to employees. For each of the following categories, please provide a brief summary of the company’s telework policy, outlined below. Please include:
1. Title: "Eligibility". Which workers are eligible for telework (include specific criteria, such as tenure, employment type, job characteristics, or anything else) \n
2. Title: "Frequency". Frequency of telework allowed (such as numbers of days per week permitted and/or minimum requirements for the employee’s presence in the office) \n
3. Title: "Arrangements". Process by which telework arrangements are established (such as whether permission must be obtained and whether the manager has discretion over the allowability of telework) \n
4. Title: "Costs". Coverage of costs associated with telework (such as IT setup and meal allowance) \n
5. Title: "Privacy". Provisions for the right to disconnect and the privacy of the worker \n
In each category, cite the article number in the document where the relevant information is provided. Label each section with the requested title. Please summarize the key points in bullet points, within a 300-token limit. Adhering to the 300-token limit is crucial for this summary. Please ensure it does not exceed this length. If the document does not mention a specific topic listed here, state "the document does not mention [topic]", where [topic] is the relevant topic. \n \n
'''

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    trust_remote_code=True, 
    device_map=device, 
    #load_in_8bit=True,
    quantization_config=bnb_config
)
model.load_lora_weights("fulltext-lora-weights/model_weights", weight_name="adapter_model.safetensors")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# inference function
def respond(f):

    with open('fulltext-pilot/'+f, 'r') as file:
        file_content = file.read()

    input_text = prompt + str(file_content) + "### Assistant: "
    inputs = tokenizer(input_text, return_tensors="pt")
    
    if device != "cpu":
        inputs = inputs.to('cuda')
        
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=60, max_new_tokens=512, temperature = 0.1)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    #return output.split("### Assistant: ")[1]
    return output

# List all files in the directory
files_in_directory = os.listdir('fulltext-pilot')

# Filter out files that end with ".txt"
txt_files = [file for file in files_in_directory if file.endswith('.txt')]
print(txt_files)

responses = txt_files.map(respond)
responses.to_csv('pilot_responses.csv')
