#Author Patrik Johansson

import string
import codecs
import torch
from torch import nn
import os
from data import DataSource

def embeddings():

    #börjat och tagit inspiration från assignment 4 och 
    #https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    # i hur embeddings genomförs.

    data_path = 'data'
    #model_path = 'model.txt'
    vocab_file = 'vocab.txt'
    source = DataSource(data_path)
    word_to_idx = dict(int)
    max_len = 0
    if os.path.isfile(vocab_file):
        with open('embedding.txt', encoding = 'utf8') as f:
            for word in f:
                word_to_idx[word] = len(word_to_idx)
                if len(word) > max_len:
                    max_len = len(word)
        embeddings = nn.Embedding(len(word_to_idx), max_len)
        embed_tensor = torch.tensor([word_to_idx], dtype=torch.long)
        # if we want to se our results, the look up:
        print(embeddings(embed_tensor))
        return embed_tensor


"""

    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)



    embedding = nn.Embedding(len(vocab), dim)

    source = DataSource(data_path)
    if os.path.source()
        if os.path.isfile(model_path):

    embedding = []
    word_to_id = dict(int)

    open('embedding.txt', encoding = 'utf8') as f:
        for line in f:
        data = line.split()
        word = data[0]
        vec = [float(x) for x in data[1:]]
        embedding.append(vec)
        word_to_id[word] = len(word_to_id)
    
    open('embedding.txt', encoding = 'utf8') as f:
        
        else:
            print("error loading file")
    
    for words in vocab:
    
    word_to_id = {c:i for i,c in enumerate(CHARS)}

    D = len(embeddings[0])

    embeddings.insert(word_to_id[padding_word], [0]*D)  # <PAD> has an embedding of just zeros
    embeddings.insert(word_to_id[unknown_word], [-1]*D)      # <UNK> has an embedding of just minus-ones

    dim, word_to_id, embeddings = load_glove_embeddings('/datasets/dd2417/glove.6B.50d.txt')

print(embeddings[word_to_id['good']])

"""