from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np

start_token = 'start '             # start token
end_token = ' end'                 # end token

""" Adds start and end token to every sentence in the dataset"""
def add_se_tokens(captions_list_oflists):
    captions_with_se_tokens = [[start_token + caption + end_token
                        for caption in captions_list]
                        for captions_list in captions_list_oflists]
    
    return captions_with_se_tokens


""" Flatten list of lists to one list to extract each word in the captions dataset"""
def flatten(captions_listlist):
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]
    
    return captions_list  # flattened list


"""Wraps the Tokenizer-class from Keras to tokenie a list of lists of captions"""
class TokenizerWrap(Tokenizer):
    
    """ num_words : maximum number of words in the vocab dictionary , texts: list of captions flattened"""
    def __init__(self, texts, num_words=None):  
        
        # ovv_token: unknown word token , adds it to the vocab dictionary
        Tokenizer.__init__(self, num_words=num_words, oov_token="<unk>", 
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')  # 1 indexed
        # Create the vocabulary from the texts using keras text preprocessing functions
        self.fit_on_texts(texts)
        self.word_index['<pad>'] = 0                  # Index of padding zero
        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),self.word_index.keys()))
        
    """Returns a word given its index in the vocab dictionary"""
    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token] # return empty word if token is zero
        return word 

    """Convert a list of integer-tokens to a string."""
    def tokens_to_string(self, tokens):
        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """
        
        # Note that text_to_sequences() takes a list of texts.
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        
        return tokens
    