# import pandas as pd
# import numpy as np
# from gensim.models import FastText
# from gensim.utils import simple_preprocess
# from gensim.models.fasttext import FastText as gensim_FastText
# from gensim.models import KeyedVectors as gensim_KeyedVectors
# import gensim
# # Load the model
# def load_model(model_path):
#     # model = gensim_KeyedVectors.load_word2vec_format(model_path, binary=False)
#     # model.save_word2vec_format('fasttext_model/wiki-news-300d-1M-subword.bin', binary=True)
    
#     # lOAD .bin file
#     model = gensim_KeyedVectors.load_word2vec_format(model_path, binary=True)
#     return model

# # Preprocess the text
# def preprocess_text(text):
#     if not isinstance(text, str):
#         text = str(text)
#     return simple_preprocess(text, deacc=True)

# # convert text to vector using fasttext embeddings
# def text_to_vector(text, model):
#     words = preprocess_text(text)
#     vectors = [model[word] for word in words if word in model]
#     if not vectors:
#         return np.zeros(model.vector_size)  # Return zero vector if no words are in the model
#     return np.mean(vectors, axis=0)

# # Load the data and preprocess it
# def load_data(data_path, model):
#     data = pd.read_csv(data_path)
#     data['reviewText'] = data['reviewText'].apply(lambda x: text_to_vector(x, model))
#     return data

# # Save the preprocessed data
# def save_data(data, data_path):
#     data.to_csv(data_path, index=False)

# if __name__ == '__main__':
    
    
    
#     model_path = 'fasttext_model/wiki-news-300d-1M-subword.bin'
#     model = load_model(model_path)
#     data = load_data('data.csv', model)
    
#     save_data(data, 'data_preprocessed.csv')
    
    
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
        # pad the vectors with zeros
        vectors += [np.zeros(model.vector_size)] * (max_seq_length - len(vectors))
        
    return np.array(vectors)

def load_data(data_path, max_seq_length):
    
    # load the data
    
    print("Loading data")
    data = pd.read_csv(data_path)
    avg_length = avg_sequence_length(data)
    print(f'Average sequence length: {avg_length}')
    
    X = np.array([text_to_vector(text, max_seq_length) for text in data['reviewText']])
    y = data['overall'].values - 1
    
    # calculate the average sequence length
    
    
    
    print("Data Loading completed")
    return X, y

