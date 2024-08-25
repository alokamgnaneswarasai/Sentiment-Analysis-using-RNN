
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.models import FastText
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors as gensim_KeyedVectors
# model = api.load('fasttext-wiki-news-subwords-300')
model = gensim_KeyedVectors.load_word2vec_format('fasttext_model/wiki-news-300d-1M-subword.bin', binary=True)

def preprocess_text(text):
    
    # tokenize and preprocess the text
    if not isinstance(text, str):
        text = str(text)
    return simple_preprocess(text, deacc=True)

def avg_sequence_length(data):
    return data['reviewText'].apply(lambda x: len(str(x).split())).mean()

def text_to_vector(text,max_seq_length):
    
        
    # preprocess the text
    
    words = preprocess_text(text)
    
    # get the vectors for each word in the text
    vectors = [model[word] for word in words if word in model]
    
    
    
    if len(vectors)>max_seq_length:
        vectors = vectors[:max_seq_length]
        
    else:
        # post-pad the vectors with zeros
        # vectors += [np.zeros(model.vector_size)] * (max_seq_length - len(vectors))
        
        # pre-pad the vectors with zeros
        vectors = [np.zeros(model.vector_size)] * (max_seq_length - len(vectors)) + vectors
        
    return np.array(vectors)

def load_data(data_path, max_seq_length,label_shifting=True):
    
    # load the data
    
    print("Loading data")
    data = pd.read_csv(data_path)
    avg_length = avg_sequence_length(data)
    print(f'Average sequence length: {avg_length}')
    
    X = np.array([text_to_vector(text, max_seq_length) for text in data['reviewText']])
    
    if label_shifting:
        y = data['overall'].values - 1
        
    else:
        y = data['overall'].values
   
    
    # calculate the average sequence length
    
    
    
    print("Data Loading completed")
    return X, y

