import os
from pathlib import Path
import torch
import re
import random
import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

MAX_LEN = 64

corpus_movie_conv = '/Users/sylvesteranthony/Documents/Bert/datasets/movie_conversations.txt'
corpus_movie_lines = '/Users/sylvesteranthony/Documents/Bert/datasets/movie_lines.txt'
with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()
with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()
    
    
lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]
    
pairs = []
for con in conv:
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []
        
        if i == len(ids) - 1:
            break
        
        first = lines_dic[ids[i]].strip()
        second = lines_dic[ids[i + 1]].strip()
        
        qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
        qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)

# os.mkdir('./data')
# text_data = []
# file_count = 0

# for sample in tqdm.tqdm([x[0] for x in pairs]):
#     text_data.append(sample)
#     if len(text_data) == 10000:
#         with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
#             fp.write('\n'.join(text_data))
#         text_data = []
#         file_count += 1
        
# paths = [str(x) for x in Path('./data').glob('**/*.txt')]

# tokenizer = BertWordPieceTokenizer(
#      clean_text = True,
#      handle_chinese_chars = False,
#      strip_accents = False,
#      lowercase = True
#     )

# tokenizer.train(
#     files = paths,
#     vocab_size = 30_000,
#     min_frequency = 5,
#     limit_alphabet = 1000,
#     wordpieces_prefix = '##',
#     special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
# )

# os.mkdir('./bert-it-1')
# tokenizer.save_model('./bert-it-1','bert-it')
tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)   


class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
        
    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, item):
        t1, t2, is_next_label = self.get_sent(item)
        
        
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
         
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]
        
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2) [:self.seq_len] 
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding),bert_label.extend(padding), segment_label.extend(padding)
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label
                  }
        
        return {key:torch.tensor(value) for key,value in output.items()}
    
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []
        
        for i, token in enumerate(tokens):
            prob = random.random()
            
            token_id = self.tokenizer(token)['input_ids'][1:-1]
            
            if prob < 0.15:
                prob /= 0.15
                
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])
                        
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))
                        
                else:
                    output.append(token_id)
                    
                output_label.append(token_id)
                        
            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)
            
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label
    
    def  get_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        
        if random.random() > 0.5:
            return t1, t2, 1
        
        else:
            return t1, self.get_random_line(),0          
        
    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]
    
    def get_random_line(self):
        return self.lines[random.randrange(len(self.lines))][1]
    
train_data = BERTDataset(
    pairs, seq_len = MAX_LEN, tokenizer = tokenizer
    )

train_loader = DataLoader(
    train_data, batch_size = 32, shuffle = True, pin_memory = True
     
)

sample_data = next(iter(train_loader))

print(train_data[random.randrange(len(train_data))])

                               
        
class PositionalEmbedding(torch.nn.Module):
    
    def __init__(self, d_model, max_len = 128):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return self.pe
    
class BERTEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, seq_length = 64, dropout = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx = 0)
        self.position = PositionalEmbedding(d_model= embed_size, max_len = seq_length)
        self.dropout = torch.nn.Dropout(p = dropout)
        
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
    
class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)
        
        
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3)
        key = key.view(key.shape[0], -1, self.heads. self.d_k).permute(0,2,1,3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3)
        
        scores = torch.matmul(query, key.permute(0,1,3,2)) / math.sqrt(query.size(-1))
        
        scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim = -1)
        weights = self.dropout(weights)
        
        context = torch.matmul(weights, value)
        
        return self.output_linear(context)
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, middle_dim = 2048, dropout = 0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
        
    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out 
    
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model = 768, heads = 12, feed_forward_hidden = 768 * 4, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim = feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)
        
        
    def forward(self, embeddings, mask):
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded