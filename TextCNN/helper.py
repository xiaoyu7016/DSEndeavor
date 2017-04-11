
# coding: utf-8
import os
import re
import numpy as np


# ### Batch indices generator
# 1. Shuffle the indices
# 2. Put in batch

def generate_batch_indices(data_size, batch_size, shuffle=True):
    
    batch_indices_dict = {}
    
    num_batches = int((data_size-1)/batch_size) + 1
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1) * batch_size, data_size)  
        batch_indices_dict[batch_num] = shuffle_indices[start_index:end_index]

    return batch_indices_dict


# function to preprocess the text of the documents
def txt_to_words(raw_txt, remove_stopwords, join=False):
    
    # Remove things
    string = re.sub(r"[^A-Za-z\']", " ", raw_txt)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    
    string = string.strip().lower().split()
    
    # Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        string = [w for w in string if not w in stops]
    
    if join:
        return (" ".join(string))
    else:
        return (string)



def create_flat_vector(input_word_list, trained_wordvec, words_per_doc, wordvec_len):
    
    doc_2d = np.zeros(shape=(words_per_doc,wordvec_len))
    
    for i, word in enumerate(input_word_list):
        if i < words_per_doc:
            if word in trained_wordvec:
                doc_2d[i,] = trained_wordvec[word]
            else:
                doc_2d[i,] = 0
        else:
            #truncate
            pass
        
    return doc_2d.flatten()



def create_X_rt(input_array, trained_wordvec, words_per_doc, wordvec_len):
    """input_array has shape (x,)
       each element is a doc represented by a list of words"""
    
    X_train = np.zeros(shape = (input_array.shape[0],words_per_doc*wordvec_len))
    
    for i,word_list in enumerate(input_array):

        doc_vec = create_flat_vector(word_list, trained_wordvec, words_per_doc, wordvec_len)
        
        X_train[i,] = doc_vec
        
    return X_train

