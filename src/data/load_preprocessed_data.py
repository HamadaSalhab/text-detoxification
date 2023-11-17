import pickle
import pandas as pd

PREPROCESSED_DATASET_PATH = '../data/interim/preprocessed_seq2seq.tsv'
VOCABS_DIR = '../data/interim/'
REFERENCE_VOCAB_PATH = VOCABS_DIR + 'reference_vocab.pkl'
TRANSLATION_VOCAB_PATH = VOCABS_DIR + 'translation_vocab.pkl'

def load_vocabs():
    """
    Load the vocabularies for reference and translation from their respective pickle files.

    This function assumes that the vocabularies have been pre-processed and stored in a
    specific directory as pickle files. The paths are constructed using a predefined directory
    and the filenames 'reference_vocab.pkl' for the reference vocabulary and 
    'translation_vocab.pkl' for the translation vocabulary.

    Returns:
        tuple: A tuple containing two dictionaries:
               - reference_vocab: A dictionary mapping each reference token to its corresponding index.
               - translation_vocab: A dictionary mapping each translation token to its corresponding index.
    """
    # Load the reference vocabulary from the pickle file
    with open(REFERENCE_VOCAB_PATH, 'rb') as f:
        reference_vocab = pickle.load(f)
    
    # Load the translation vocabulary from the pickle file
    with open(TRANSLATION_VOCAB_PATH, 'rb') as f:
        translation_vocab = pickle.load(f)
    
    # Return both vocabularies as a tuple
    return reference_vocab, translation_vocab

def get_dataframe(sample_ratio=1):
    """
    Retrieves a DataFrame containing the preprocessed data for sequence-to-sequence modeling.

    This method loads the preprocessed data from a tab-separated values (TSV) file, with an option to
    sample a subset of the data. This can be useful for quick testing or when working with very large
    datasets where utilizing the full dataset might be resource-intensive.

    Args:s
        sample_ratio (float, optional): A value between 0 and 1 indicating the fraction of the data
                                        to sample. A value of 1 means the entire dataset is used.
                                        Defaults to 1.

    Returns:
        pd.DataFrame: A pandas DataFrame with the preprocessed data. If `sample_ratio` is less than 1,
                      the returned DataFrame will be a randomly sampled subset of the original data.

    Raises:
        ValueError: If `sample_ratio` is not between 0 and 1.
        FileNotFoundError: If the TSV file located at `PREPROCESSED_DATASET_PATH` does not exist.

    Examples:
        # Get the full preprocessed dataset as a DataFrame
        full_data = get_dataframe()

        # Get a random 10% sample of the preprocessed dataset as a DataFrame
        sampled_data = get_dataframe(sample_ratio=0.1)
    """
    if not 0 <= sample_ratio <= 1:
        raise ValueError("Sample ratio must be between 0 and 1.")

    # Attempt to load the preprocessed dataset from the TSV file
    try:
        df = pd.read_csv(PREPROCESSED_DATASET_PATH, delimiter='\t')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The data file was not found at the specified path: {PREPROCESSED_DATASET_PATH}") from e

    # If sampling is requested, sample a fraction of the DataFrame based on the given ratio
    if sample_ratio < 1:
        df = df.sample(frac=sample_ratio, random_state=42)
        
    return df
