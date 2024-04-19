from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import pandas as pd
from datetime import datetime
import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel

hf_token = "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"
write_token = "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"

cache_dir = "/vast/palmer/scratch/odea/das293/huggingface/"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

task = 'summary'
device = "auto"

# Path to the LoRA weights exported from Predibase
if task=='summary':
    weights = "fulltext-lora-weights/model_weights"
if task=='compare':
    weights = "telework-compare-mixtral/model_weights"   

# fine-tuned linewise model
base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"        

# Prompt should be in this style due to how the data was created
#prompt = "#### Human: What is the capital of Australia?#### Assistant:"

bnb_config = BitsAndBytesConfig(
        #load_in_4bit=True,
        quantization_bits=4,
        #bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        #bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    device_map = 'auto',
    token = hf_token,
    #device_map=device#, 
    #load_in_8bit=True,
    #quantization_config=bnb_config
)

model = PeftModel.from_pretrained(model, weights)
model = model.merge_and_unload()
'''
# push model to hub
if task=='summary':
    model.push_to_hub("danascott329/mixtral-document-summaries-telework", token = write_token)
if task=='compare':
    model.push_to_hub("danascott329/mixtral-telework-compare", token = write_token)
'''
#model.load_lora_weights("fulltext-lora-weights/model_weights", weight_name="adapter_model.safetensors")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# helper functions
def respond(f):

    with open('fulltext-pilot/'+f, 'r') as file:
        file_content = file.read()

    input_text = prompt + str(file_content) + "### Assistant: "
    inputs = tokenizer(input_text, return_tensors="pt")
    
    if device != "cpu":
        inputs = inputs.to('cuda')
        
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=60, max_new_tokens=256, temperature = 0.1)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    #return output.split("### Assistant: ")[1]
    return output

def policy_compare(policy1, policy2):

    input_text = prompt + "\n\n Policy 1: \n\n" + str(policy1) + "\n\n Policy 2:\n" + str(policy2) + "### Assistant: "
    inputs = tokenizer(input_text, return_tensors="pt")
    
    if device != "cpu":
        inputs = inputs.to('cuda')
        
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=60, max_new_tokens=256, temperature = 0.1)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    #return output.split("### Assistant: ")[1]
    return output

def generate_pairs_with_indices(docs, num_pairs):
    pairs = pd.DataFrame({'doc1': [], 'doc2': [], 'policy1': [], 'policy2': []})
    for i in range(num_pairs):
        samp = docs.sample(n=2)
        doc1 = samp['doc'].iloc[0]
        doc2 = samp['doc'].iloc[1]
        policy1 = samp[topic].iloc[0]
        policy2 = samp[topic].iloc[1]
        pair = {'doc1' : doc1, 'doc2': doc2, 'policy1': policy1, 'policy2': policy2}
        #pairs = pairs.append(pair, ignore_index = True)
        pairs = pd.concat([pairs, pd.DataFrame([pair])], ignore_index=True)
    return(pairs)

start_time = datetime.now()
print('Start time: {}'.format(start_time))
'''
# inference function and prompt definition -- define for the different tasks
if task == 'summary':

    prompt = f'''
    Your task is to summarize the company’s telework policy in a way that is understandable to employees. For each of the following categories, please provide a brief summary of the company’s telework policy, outlined below. Please include:
    1. Title: "Eligibility". Which workers are eligible for telework (include specific criteria, such as tenure, employment type, job characteristics, or anything else) \n
    2. Title: "Frequency". Frequency of telework allowed (such as numbers of days per week permitted and/or minimum requirements for the employee’s presence in the office) \n
    3. Title: "Arrangements". Process by which telework arrangements are established (such as whether permission must be obtained and whether the manager has discretion over the allowability of telework) \n
    4. Title: "Costs". Coverage of costs associated with telework (such as IT setup and meal allowance) \n
    5. Title: "Privacy". Provisions for the right to disconnect and the privacy of the worker \n
    In each category, cite the article number in the document where the relevant information is provided. Label each section with the requested title. Please summarize the key points in bullet points, within a 300-token limit. Adhering to the 300-token limit is crucial for this summary. Please ensure it does not exceed this length. If the document does not mention a specific topic listed here, state "the document does not mention [topic]", where [topic] is the relevant topic. \n \n
    '''
    '''
    # List all files in the directory
    files_in_directory = os.listdir('fulltext-pilot')

    # Filter out files that end with ".txt"
    txt_files = [file for file in files_in_directory if file.endswith('.txt')]
    print(txt_files)

    responses = list(map(respond, txt_files))

    df = pd.DataFrame({
        'doc': txt_files,
        'summary': responses
    })

    filename = 'pilot_responses.csv'
    '''

if task == 'compare':
    '''
    # Load list of potential policies to compare
    docs = pd.concat(
        [pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v0320.csv'), 
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v2_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v3_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v4_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v5_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v6_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v7_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v8_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v9_v0320.csv'),
        pd.read_csv('wide-files/telework_pilot_summary-mixtral-finetuned_wide_v10_v0320.csv')], 
        ignore_index=True
        )
    '''
    # set topic
    topic = "Frequency"

    if topic == "Frequency":
        prompt = f'''
        Policy 1 and Policy 2 below describe the frequency of telework allowed at two different companies. Which company's policy is more generous? Please format your response as follows: \n
        1. Policy 1, Policy 2, or uncertain \n
        2. confidence (0-1) \n
        3. explanation (less than 25 words) \n
        If either of the two policies does not mention frequency, state "Policy [1,2] does not mention frequency" for the relevant policy and do not answer the question. \n
        '''
    if topic == "Eligibility":
        prompt = f'''
        Policy 1 and Policy 2 below describe the eligibility requirements for telework at two different companies. Which company's policy is more generous? Please format your response as follows: \n
        1. Policy 1, Policy 2, or uncertain \n
        2. confidence (0-1) \n
        3. explanation (less than 25 words) \n
        If either of the two policies does not mention eligibility, state "Policy [1,2] does not mention eligibility" for the relevant policy and do not answer the question. \n
        '''
    if topic == "Arrangements":
        prompt = f'''
        Policy 1 and Policy 2 below describe the process for arranging telework at two different companies. Which company's policy is more favorable to the worker? Please format your response as follows: \n
        1. Policy 1, Policy 2, or uncertain \n
        2. confidence (0-1) \n
        3. explanation (less than 25 words) \n
        If either of the two policies does not mention this topic, state "Policy [1,2] does not mention arrangements" for the relevant policy and do not answer the question. \n
        '''
    if topic == "Costs":
        prompt = f'''
        Policy 1 and Policy 2 below describe the policy for covering costs associated with telework at two different companies. Which company's policy is more generous? Please format your response as follows: \n
        1. Policy 1, Policy 2, or uncertain \n
        2. confidence (0-1) \n
        3. explanation (less than 25 words) \n
        If either of the two policies does not mention costs, state "Policy [1,2] does not mention costs" for the relevant policy and do not answer the question. \n
        '''
    if topic == "Privacy":
        prompt = f'''
        Policy 1 and Policy 2 below describe the provisions for protecting the worker's privacy and/or their right to disconnect while teleworking at two different companies. Which company's policy is more favorable to the worker? Please format your response as follows: \n
        1. Policy 1, Policy 2, or uncertain \n
        2. confidence (0-1) \n
        3. explanation (less than 25 words) \n
        If either of the two policies does not mention privacy or the right to disconnect, state "Policy [1,2] does not mention privacy" for the relevant policy and do not answer the question. \n
        '''
    '''
    df = generate_pairs_with_indices(docs, 50)
    print(df)

    df['output'] = df.apply(lambda row: policy_compare(row['policy1'],row['policy2']), axis = 1)
    print(df)
    filename = 'wide-files/'+ topic + '_compare_combined_mixtral_pilot.csv'
    '''

'''
df.to_csv(filename)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
