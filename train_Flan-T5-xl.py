from datasets import load_dataset, concatenate_datasets
import numpy as np
from transformers import AutoTokenizer
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Loading Model and Tokenizer # 加载模型和分词器
model_id = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_id)

from datasets import load_from_disk, DatasetDict
# Loading dataset # 加载数据集
tokenized_train_dataset = load_from_disk("data/train")
tokenized_test_dataset = load_from_disk("data/eval")

# Creating a dataset dictionary # 创建数据集字典
tokenized_dataset = DatasetDict({
    'train': tokenized_train_dataset,
    'test': tokenized_test_dataset
})

print(tokenized_dataset)

from transformers import AutoModelForSeq2SeqLM


model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")


from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# 定义LoRA配置
lora_config = LoraConfig(
 r=16,  # Rank in LoRA, used to control the number of parameters of the LoRA adapter # LoRA中的秩（rank），用于控制LoRA适配器的参数数量
 lora_alpha=32,  # The alpha value in LoRA is used to adjust the expansion factor of the LoRA adapter # LoRA中的alpha值，用于调整LoRA适配器的扩张因子
 target_modules=["q", "v"],  # Specifying the target modules to which the LoRA adapter is to be applied, here the "q" (query) and "v" (value) modules # 指定要应用LoRA适配器的目标模块，这里是“q”（查询）和“v”（值）模块
 lora_dropout=0.05,  # Dropout rate in LoRA adapter to prevent overfitting # LoRA适配器中的dropout率，用于防止过拟合
 bias="none",  
 task_type=TaskType.SEQ_2_SEQ_LM  #Specifying the task type, here is a sequence-to-sequence language model #  指定任务类型，这里是序列到序列的语言模型
)
# Preparing the int-8 model for training # 为训练准备int-8模型
model = prepare_model_for_int8_training(model)

# Adding LoRA Adapter # 添加LoRA适配器
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Trainable parameters: 18874368 || Total parameters: 11154206720 || Trainable ratio: 0.16921300163961817 # 可训练参数: 18874368 || 所有参数: 11154206720 || 可训练比例: 0.16921300163961817

from transformers import DataCollatorForSeq2Seq

# Defining a pad token id for a tag to ignore the pad token of the tokenizer when calculating the loss # 定义一个标签的pad token id，用于在计算损失时忽略tokenizer的pad token
label_pad_token_id = -100

# Creating a data collator to prepare data batches for model training or evaluation # 创建数据整理器（Data collator），用于准备模型训练或评估时的数据批次
data_collator = DataCollatorForSeq2Seq(
    tokenizer,  
    model=model, 
    label_pad_token_id=label_pad_token_id,  # Setting label pad token id
    pad_to_multiple_of=8
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

output_dir="lora-flan-t5-xl"

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  
	auto_find_batch_size=64, 
    learning_rate=1e-3,
    num_train_epochs=3,  
    logging_dir=f"{output_dir}/logs", 
    logging_strategy="steps",  
    logging_steps=500,  
    save_strategy="no",  # Setting not to save the model during training # 设置不在训练过程中保存模型
    report_to="tensorboard",  # Setting report output to tensorboard # 设置报告输出到tensorboard
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args, 
    data_collator=data_collator, 
    train_dataset=tokenized_dataset["train"], 
)
model.config.use_cache = False  # Disable caching to avoid warnings, but re-enable when doing inference # 禁用缓存以避免警告，但请在进行推理时重新启用

# start training
trainer.train()

# Saving the LoRA model and tokenizer # 保存LoRA模型和tokenizer
peft_model_id="results"  
trainer.model.save_pretrained(peft_model_id)  
tokenizer.save_pretrained(peft_model_id)  
# If you want to save the base model, you can call the following code # 如果你想保存基础模型，可以调用以下代码
# trainer.model.base_model.save_pretrained(peft_model_id)
