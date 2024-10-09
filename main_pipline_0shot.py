from get_isSATD import get_isSATD_with_0_or_1
from query_with_nl import query_with_nl
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm


def get_prompt(text, task):
    """ prompt example:
    part1: Self-admitted debts have eight common types : Architecture, Build, Code, Defect, Design, Documentation, Requirements, Test.
           Self-admitted debts have four common sources : code-comments, issues, pull-requests, commit-messages.
    part2: Tell me which of the eight types the following technical debt belongs to?
    part3: ### Technical debt text: (From: 1code-comments) As we don't use the CxfSoap component anymore, it's time to clean it up.
    prompt = part1 + part2 + part3
    """
    context = ""
    with open('Pipeline/prompt_0shot.txt', 'r') as f:
        for line in f:
            context += line
    context = context.strip()
    context_sample  = "\nTell me which of the eight types the following technical debt belongs to?\n"
    instruction = context + context_sample
    filenames = ['1code-comments', '2issues', '3pull-requests', '4commit-messages']
    file_name = filenames[task - 1]
    context2 = f"### Technical debt text: (From: {file_name}){text}\n"
    query = instruction + context2
    
    return query

# 得到结果by一条text
def pipline(text, text_pro, task, glm_tokenizer, glm_model):
    lable1 = get_isSATD_with_0_or_1(text_pro, task)
    if lable1 == 1:
        prompt = get_prompt(text, task)
        response = query_with_nl(glm_tokenizer, glm_model, prompt)
        return response
    elif lable1 == 0:
        return 'NON-SATD'
    else:
        raise KeyError("label is error")

#================可变参数=====================
model_name = "satd-glm4-9b-chat-sft"
dev_pro_path = 'Pipeline/dev_data/dev_1code-comments_process.csv'
dev_path = 'Pipeline/dev_data/dev_1code-comments.csv'
output_dir = 'Pipeline/pipline_0shot_1.csv'
task_id = 1
device = "cuda:1"
#=============================================


#================模型和数据准备================
glm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
glm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


#================访问两个模型，输出结果====================
res_list = []
df_dev_pro = pd.read_csv(dev_pro_path)
df_dev = pd.read_csv(dev_path)
for i, row in tqdm(df_dev_pro.iterrows(), total=df_dev_pro.shape[0]):
    # task = row['task']
    # task_id = int(task[0])
    index = row['index']
    fr = row['from']
    cls = row['class']
    text = df_dev[df_dev['index'] == index]['text'].values
    text = text[0]
    text_pro = row['text']
    response = pipline(text, text_pro, task_id, glm_tokenizer, glm_model)
    lst_tmp = [index, fr, text_pro, cls, response]
    print(lst_tmp)
    res_list.append(lst_tmp)
    
df_res = pd.DataFrame(res_list, columns=['index', 'from', 'text_pro', 'class', 'predict'])
df_res.to_csv(output_dir)   

