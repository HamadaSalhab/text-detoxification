import argparse
from collections import Counter
from itertools import chain
from make_dataset import get_dataset
import pandas as pd
from preprocess import build_vocab, tokenize_and_numericalize


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Preprocess dataset for Seq2Seq detoxification.")
    parser.add_argument('--save_path', type=str, default='./data/interim/preprocessed_seq2seq.tsv',
                        help='Path to save the preprocessed dataset.')
    
    args = parser.parse_args()
    
    
    df = get_dataset()

    print('Preprocessing...')
    
    # Tokenize texts
    reference_tokenized = df['reference'].str.split().tolist()
    translation_tokenized = df['translation'].str.split().tolist()

    # Build vocabularies
    reference_vocab = build_vocab(reference_tokenized)
    translation_vocab = build_vocab(translation_tokenized)

    # Numericalize texts
    df['reference_numericalized'] = df['reference'].apply(lambda x: tokenize_and_numericalize(x, reference_vocab))
    df['translation_numericalized'] = df['translation'].apply(lambda x: tokenize_and_numericalize(x, translation_vocab))
    print('Finished preprocessing.')

    # Save preprocessed data
    print('saving the file to ' + args.save_path)
    df[['reference_numericalized', 'translation_numericalized']].to_csv(args.save_path, sep='\t', index=False)
    print('Processed data has been saved successfully to ' + args.save_path)
    