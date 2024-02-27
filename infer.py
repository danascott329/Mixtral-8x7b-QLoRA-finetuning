from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import pandas as pd





device = "auto"
#model_path = "outputs/merged_model"             # Path to the combined weights
model_path = "danascott329/mixtral-ft-linewise"        # fine-tuned linewise model

# Prompt should be in this style due to how the data was created
#prompt = "#### Human: What is the capital of Australia?#### Assistant:"

prompt = f''' ### Instruction:
Your task is to summarize the company’s telework policy in a way that is understandable to employees. The following document excerpt is taken from a larger policy document. First, identify which of the following categories are discussed in the document. If none are, please say "no categories applicable." For each applicable category, please provide a brief summary of the company’s telework policy, outlined below. For any categories that are not applicable, state "not applicable". The categories are: \n
1. Title: "Eligibility". Which workers are eligible for telework (include specific criteria, such as tenure, employment type, job characteristics, or anything else) \n
2. Title: "Frequency". Frequency of telework allowed (such as numbers of days per week permitted and/or minimum requirements for the employee’s presence in the office) \n
3. Title: "Arrangements". Process by which telework arrangements are established (such as whether permission must be obtained and whether the manager has discretion over the allowability of telework) \n
4. Title: "Costs". Coverage of costs associated with telework (such as IT setup and meal allowance) \n
5. Title: "Privacy". Provisions for the right to disconnect and the privacy of the worker \n
In each category, cite the article number in the document where the relevant information is provided. Label each section with the requested title. Please summarize the key points in bullet points, within a 50-token limit per category. Adhering to the 50-token limit per category is crucial for this summary. Please ensure it does not exceed this length. \n \n
### Input: 
'''



bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map=device, 
    #load_in_8bit=True,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# inference function
def respond(item):

    input_text = prompt + str(item) + "### Assistant: "
    inputs = tokenizer(input_text, return_tensors="pt")
    
    if device != "cpu":
        inputs = inputs.to('cuda')
        
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=60, max_new_tokens=512, temperature = 0.1)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output.split("### Assistant: ")[1]

# apply to a validation sample
sample = pd.read_csv('lines_autoflagged_sections_1_telework_primary.csv')[:10]
outs = sample['full'].map(respond)

outs.to_csv('pilot.csv')
