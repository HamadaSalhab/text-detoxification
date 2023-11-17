from collections import Counter
from itertools import chain

# Define special tokens that will be used in the data preprocessing
PAD_TOKEN = '<pad>'  # Token used for padding sentences to the same length
SOS_TOKEN = '<sos>'  # Start-of-sentence token
EOS_TOKEN = '<eos>'  # End-of-sentence token

def build_vocab(texts, min_freq=2):
    """
    Build a vocabulary dictionary for text tokenization.
    
    This function takes in a list of tokenized texts (lists of strings), counts the frequency
    of each token across all texts, and creates a dictionary mapping each token to a unique
    integer index. Only tokens that appear at least 'min_freq' times are included in the
    vocabulary.

    Args:
        texts (list of list of str): Tokenized texts as a list of lists of tokens.
        min_freq (int): Minimum frequency for a token to be included in the vocabulary.

    Returns:
        dict: A dictionary with tokens as keys and their corresponding unique integer index as values.
    """
    # Count the frequencies of each token across the texts
    counter = Counter(chain.from_iterable(texts))
    
    # Create the vocabulary by assigning an index to each token based on its frequency
    # and ensuring it meets the minimum frequency requirement
    vocab = {token: idx for idx, (token, freq) in enumerate(counter.items()) if freq >= min_freq}
    
    # Include special tokens in the vocabulary with predefined indices
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, **vocab}
    
    return vocab

def tokenize_and_numericalize(text, vocab):
    """
    Convert a text to a list of numerical indices based on the provided vocabulary.
    
    Each token in the text is replaced by its corresponding index from the vocabulary.
    Special tokens for the start and end of the sentence are added to the numericalized list.
    
    Args:
        text (str): A string of text to be tokenized and numericalized.
        vocab (dict): A dictionary mapping tokens to their numerical indices.

    Returns:
        list of int: A list of numerical indices representing the tokenized text.
    """
    # Tokenize the text by splitting on whitespace
    tokens = text.split()
    
    # Convert tokens to their corresponding indices from the vocabulary
    # with special tokens for the beginning and end of the sentence
    numericalized = [vocab[SOS_TOKEN]] + \
                    [vocab.get(token, vocab[PAD_TOKEN]) for token in tokens] + \
                    [vocab[EOS_TOKEN]]
    
    return numericalized