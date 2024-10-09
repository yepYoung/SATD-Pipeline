import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score
import sentence_transformers

def get_the_most_relevant_items_for_an_item(query_embed, keys_embed, k):
  cos_sim = sentence_transformers.util.cos_sim(query_embed, keys_embed)
  itemId_similarity = dict(zip(range(len(keys_embed)),cos_sim.tolist()[0]))
  itemId_similarity = dict(sorted(itemId_similarity.items(), key=lambda item: item[1], reverse=True)) # sort
  itemId_similarity = [(idx, itemId_similarity[idx]) for idx in list(itemId_similarity)[:k]] # take top k
  return itemId_similarity 

def get_Ksample(train_data_embeds, text,k, st_model, text_list, class_list):
    text_data_embeds = st_model.encode(text)
    tups = get_the_most_relevant_items_for_an_item(text_data_embeds, train_data_embeds, k)
    tup_list = []
    for tup in tups:
        (i, t) = tup
        tup_list.append((text_list[i], class_list[i]))
    
    return tup_list


def get_prompt_fewshot(train_data_embeds, text, task, k, st_model, text_lists, class_lists):
    """prompt example:
    part1: Self-admitted debts have eight common types : Architecture, Build, Code, Defect, Design, Documentation, Requirements, Test.
           Self-admitted debts have four common sources : code-comments, issues, pull-requests, commit-messages.
           Here are some examples:
            { ### Label:Design    ### Technical debt text: (From:2-issues) text_sample}
            { ### Label:Design    ### Technical debt text: (From:2-issues) text_sample}
            { ### Label:Design    ### Technical debt text: (From:2-issues) text_sample}
    part2: Tell me which of the eight types the following technical debt belongs to?
    part3: ### Technical debt text: (From: 1code-comments) As we don't use the CxfSoap component anymore, it's time to clean it up.
    prompt = part1 + part2 + part3
    """
    text_list = text_lists[task - 1]
    class_list = class_lists[task - 1]
    train_data_embeds = train_data_embeds[task - 1]
    prompt1 = ''
    with open('Pipeline/prompt_fewshot.txt', 'r') as f:
        for line in f:
            prompt1 += line
    prompt1 = prompt1.strip()
    prompt1 += '\n'
    
    filenames = ['1code-comments', '2issues', '3pull-requests', '4commit-messages']
    file_name = filenames[task - 1]
    sample_list = get_Ksample(train_data_embeds, text, k, st_model, text_list, class_list)
    prompt2 = ''
    for text_sample, label in sample_list:
        prompt2_part1 = '{' + f'### Label:{label}'
        prompt2_part2 = f'   ### Technical debt text: (From:{file_name}){text_sample}' + '}\n'
        prompt2 += prompt2_part1 + prompt2_part2
    
    query = "\nTell me which of the eight types the following technical debt belongs to?\n"
   
    context2 = f"### Technical debt text: (From: {file_name}){text}\n"
    
    prompt = prompt1 + prompt2 + query + context2
    
    return prompt

print('.......')
# 获取k采样需要的数据
st_model = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')
print('//////')
knn_tarin_data_path_1 = 'LM-classs-prompt/train_dev/train_1code-comments.csv'
knn_tarin_data_path_2 = 'LM-classs-prompt/train_dev/train_2issues.csv'
knn_tarin_data_path_3 = 'LM-classs-prompt/train_dev/train_3pull-requests.csv'
knn_tarin_data_path_4 = 'LM-classs-prompt/train_dev/train_4commit-messages.csv'
knn_tarin_data_pathes = [knn_tarin_data_path_1, knn_tarin_data_path_2, knn_tarin_data_path_3, knn_tarin_data_path_4]
text_lists = []
class_lists = []
train_data_embeds_set = []
for i in range(4):
    knn_tarin_data_path = knn_tarin_data_pathes[i]
    df = pd.read_csv(knn_tarin_data_path)
    text_list = []
    class_list = []
    for index, row in df.iterrows():
        text_list.append(row['text'])
        class_list.append(row['class'])  
    print('..............')
    train_data_embeds = st_model.encode(text_list, show_progress_bar=True)
    text_lists.append(text_list)
    class_lists.append(class_list)
    train_data_embeds_set.append(train_data_embeds)





device = 'cuda:0'
model_name = "satd-glm4-9b-chat-sft"
glm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
glm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
    # device_map='auto'
).to(device).eval()
gen_kwargs = {"max_length": 24000, "do_sample": True, "top_k": 1}

k=3

df = pd.read_csv('Pipeline/dev_data/Design.csv')
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    text = row['text']
    source = row['source']
    task_id = int(source[0])
    prompt = get_prompt_fewshot(train_data_embeds_set, text, task_id, k, st_model, text_lists, class_lists)
    inputs = glm_tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                add_generation_prompt=True,
                                tokenize=True,
                                return_tensors="pt",
                                return_dict=True
                                )
    inputs = inputs.to(device)
    response1 = ''
    with torch.no_grad():
        outputs = glm_model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response1 = glm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response1 = response1.replace('\n', ' ')
        print(response1)
    with open('Pipeline/dev_data/design_res_knn.txt', 'a') as f:
        f.write(response1) 
        f.write('\n')
        