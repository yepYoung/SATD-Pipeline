# Identifying and Classifying Multiple Sourced and Categorized Self-Admitted Technical Debts: A Pipeline
---
This  repository contains the models, dataset and experimental code mentioned in the paper. Specifically, experimental code includes the implementation code of our pipeline and the process of training, the dataset includes the preprocessed complete dataset used for training and testing, and models includes our models in pipeline and RQ1-3.
---
#### dataset and models

| Dataset&Models| description | Link |
| ----------- | ----------- | --------- |
| Dataset-satd_aug | Total augmented dataset including all SATD sentence, instruction and categories. | [Link1](https://huggingface.co/datasets/chaos1203/satd_aug) |
| Model-glm4-9b-chat-sft-9class | Model training to classify all SATD sentences into 9 categories. | [Link2](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft-9class) |
| Model-glm4-9b-chat-sft | Model training to apply in pipeline to classify isSATD sentences into 8 categories. | [Link3](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft) |
| Model-satd-glm4-9b-chat-sft-noaug | Model training to classify isSATD sentences into 8 categories with data not augmented. | [Link4](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft-noaug) |
| Model-MT-Bert | Model traing to identify all SATD sentences into 2 categories(isSATD or nonSATD) | [Link4](https://box.nju.edu.cn/f/a9455caeac9547159ff2/?dl=1) |

#### code
The following is an introduction to code to make it easier for readers to use.
> The files `main_pipeline_0shot.py` and `main_pipeline_fewshot.py` are the main files to run our pipeline. And The others are tools which used by the two files. The folder `bert_config` contents the related configuration of our bert model. The folder `train_bert` contents the training details of our bert model.



