import argparse
import re
import string
import fasttext
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch import autograd
import os
import pandas as pd

# Define constants
DEF_COMMENT = 'code-comments'
DEF_COMMIT = 'commit-messages'
DEF_PULL = 'pull-requests'
DEF_ISSUE = 'issues'
DEF_MAPPING = {DEF_ISSUE: 0, DEF_COMMIT: 1, DEF_COMMENT: 2, DEF_PULL: 3}
DEF_LABELS = ['NON-SATD', 'Architecture', 'Build', 'Code', 'Defect', 'Design', 'Documentation', 'Requirement', 'Test']

class TextCNNMultitask(nn.Module):
    """
    Text CNN multitask network based on Kim CNN structure
    """

    def __init__(self, params):
        """
        Init function

        @param params:
        """
        super(TextCNNMultitask, self).__init__()
        self.params = params

        # hyper-parameters
        self.list_conv = nn.ModuleList([nn.Conv2d(1, self.params.kernel_num, (size, self.params.embed_dim))
                                        for size in self.params.kernel_sizes])
        self.dropout = nn.Dropout(params.dropout)
        self.fcs = nn.ModuleList([nn.Linear(len(self.params.kernel_sizes) * self.params.kernel_num, len(DEF_LABELS))
                                  for _ in range(len(DEF_MAPPING.keys()))])

    def forward(self, x):
        """
        Forward function

        @param x:   x shape is (batch_size, words, embed_dim)
        @return:
        """
        # x_embed shape is (batch_size, 1, number_of_words, embed_dim)
        x_embed = x.unsqueeze(1)
        # x_conv shape is (batch_size, kernel_num, feature_num, 1)
        x_list_conv = [func.relu(conv(x_embed)).squeeze(3) for conv in self.list_conv]
        # x_max_pool shape is (batch_size, kernel_num, 1)
        x_list_max_pool = [func.max_pool1d(x_conv, x_conv.size(2)).squeeze(2) for x_conv in x_list_conv]
        # x_concatenate and x_dropout shape is (batch_size, kernel_num * number_of_different_size_kernel)
        x_concatenate = torch.cat(x_list_max_pool, 1)
        x_dropout = self.dropout(x_concatenate)
        # x_list_logit shape is [output_num: (batch_size, class_num)]
        x_list_logit = [fc(x_dropout) for fc in self.fcs]

        return x_list_logit



class SATDDataset(Dataset):
    def __init__(self, data, params, embedding_model):
        self.data = data
        self.params = params
        self.embedding_model = embedding_model
        self.tokenizer = nltk.TweetTokenizer()
        self.punctuation = string.punctuation.replace('!', '').replace('?', '')

        self.label_mapping = {label: idx for idx, label in enumerate(DEF_LABELS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        task_type = item['type']
        text = self.preprocess_text(text)
        text_vector = self.text_to_vector(text)
        text_vector = torch.tensor(text_vector, dtype=torch.float32)  # Converting to numeric type # 转换为数值类型
        label = self.label_mapping[label]  # Using Value Mapping # 使用数值映射
        label = torch.tensor(label, dtype=torch.long)  # Converting to numeric type
        task_type = DEF_MAPPING[task_type]  # Using Value Mapping
        return text_vector, label, task_type

    def preprocess_text(self, text):
        text = re.sub('(//)|(/\\*)|(\\*/)', '', text).lower()
        tokens_sentences = [self.tokenizer.tokenize(t) for t in nltk.sent_tokenize(text)]
        processed_tokens_sentences = []
        for sentence in tokens_sentences:
            processed_tokens = []
            for token in sentence:
                if token == 'non-nls' or token == '<pad>':
                    processed_tokens.append(token)
                    continue
                elif token == ',':
                    processed_tokens.append('.')
                elif ' ' in token or '.' in token or '#' in token or '_' in token or '/' in token:
                    continue
                elif '>' in token or '<' in token or '@' in token:
                    continue
                elif any(char.isdigit() for char in token):
                    continue
                else:
                    processed_tokens.append(token)
            if len(processed_tokens) > 0:
                if processed_tokens[-1] != '.':
                    processed_tokens.append('.')
            processed_tokens_sentences.append(processed_tokens)
        tokens_sentences = processed_tokens_sentences
        tokens = [word for t in tokens_sentences for word in t]
        stripped = [word for word in tokens if word and (word not in self.punctuation and ':' not in word and '=' not in word and ')' not in word and '(' not in word)]
        return stripped

    def text_to_vector(self, text):
        embed_text = []
        for word in text:
            word_embed = self.embedding_model[word]
            embed_text.append(word_embed)
        embed_text = embed_text + [self.embedding_model['<pad>']] * (self.params.max_len - len(embed_text))
        return embed_text[:self.params.max_len]  # Making sure the length of the text vector does not exceed # 确保文本向量的长度不超过 max_len




def save_model(model, path, epoch):
    """
    Save the model to the specified path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, f'model_epoch_{epoch}.pt'))

def train_model(params, train_data, val_data):
    embedding_model = fasttext.load_model(params.embed_vectors)
    train_dataset = SATDDataset(train_data, params, embedding_model)
    val_dataset = SATDDataset(val_data, params, embedding_model)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)

    model = TextCNNMultitask(params)
    if params.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(params.epochs):
        model.train()
        for texts, labels, task_types in train_loader:
            if params.cuda:
                texts, labels, task_types = texts.cuda(), labels.cuda(), task_types.cuda()
            optimizer.zero_grad()
            outputs = model(texts)
            loss = 0
            for i, task_type in enumerate(task_types):
                output = outputs[task_type]
                loss += criterion(output, labels[i])
            loss.backward()
            optimizer.step()
        
        val_loss = evaluate_model(model, val_loader, criterion, params)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')

        # Save the model at the end of each epoch
        save_model(model, params.save_path, epoch + 1)

    # Save the final model
    save_model(model, params.save_path, 'final')

def evaluate_model(model, val_loader, criterion, params):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, labels, task_types in val_loader:
            if params.cuda:
                texts, labels, task_types = texts.cuda(), labels.cuda(), task_types.cuda()
            outputs = model(texts)
            loss = 0
            for i, task_type in enumerate(task_types):
                output = outputs[task_type]
                loss += criterion(output, labels[i])
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main(args):
    # Loading your data here and split into training and validation sets
    train_data_out = []
    val_data_out = []
    # Loading train.csv and dev.csv from csv as training set and validation set respectively, where text is the text and class is the label # 从csv中分别加载train.csv和dev.csv作为训练集和验证集，其中的text作为文本，class作为标签
    train_data = pd.read_csv('dataset/train.csv')
    for index, row in train_data.iterrows():
        text = row['text']
        label = row['class']
        task = row['source'][1:]
        train_data_out.append({'text': text, 'label': label, 'type': task})
    val_data = pd.read_csv('dataset/dev.csv')
    for index, row in val_data.iterrows():
        text = row['text']
        label = row['class']
        task = row['source'][1:]
        val_data_out.append({'text': text, 'label': label, 'type': task})

    train_model(args, train_data_out, val_data_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TextCNN model for SATD classification')
    parser.add_argument('--embed_vectors', type=str, default='fasttext_issue_300.bin', help='Path to the FastText embedding vectors')
    parser.add_argument('--embed_dim', type=int, default=128, help='Dimension of embedding vectors') 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--kernel_num', type=int, default=100, help='Number of kernels per convolutional layer')
    parser.add_argument('--kernel_sizes', type=list, default=[3, 4, 5], help='List of kernel sizes for convolutional layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length of input sequences')
    parser.add_argument('--cuda', action='store_true', default='CUDA', help='Use CUDA for training')
    parser.add_argument('--device', type=int, default=0, help='CUDA device to use')
    parser.add_argument('--save_path', type=str, default='./saved_models', help='Path to save the trained model')
    args = parser.parse_args()
    main(args)
    
