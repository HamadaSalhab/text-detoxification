import json
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from model_utils import TextDetoxEncoder, TextDetoxDecoder, TextDetoxSeq2SeqModel, train_detox_model, evaluate_detox_model
import argparse
import os, sys
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from model_utils import create_model


def main(args):
     
    PREPROCESSED_DATASET_PATH = './data/interim/preprocessed_seq2seq.tsv'
    MODEL_PATH = './models/SEQ2SEQ_LSTMs.pth'
    
    df = pd.read_csv(PREPROCESSED_DATASET_PATH, delimiter='\t')
    
    from sklearn.model_selection import train_test_split

    # Train/Test Split
    train, eval = train_test_split(df, test_size=0.2, shuffle=False)
    
    class TextDetoxDataset(Dataset):
        def __init__(self, dataframe):
            def string_to_list(s):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    return [int(item) for item in s.strip('[]').split(', ')]

            self.reference_numericalized = dataframe['reference_numericalized'].apply(string_to_list)
            self.translation_numericalized = dataframe['translation_numericalized'].apply(string_to_list)

        def __len__(self):
            return len(self.reference_numericalized)

        def __getitem__(self, idx):
            input_ids = self.reference_numericalized.iloc[idx]
            labels = self.translation_numericalized.iloc[idx]

            return {
                "reference": input_ids,
                "translation": labels
            }
    
    train_dataset, eval_dataset = TextDetoxDataset(train), TextDetoxDataset(eval)
    
    def text_detox_collate_fn(batch):
        reference = pad_sequence([torch.tensor(item["reference"], dtype=torch.long) for item in batch],
                                    batch_first=True, padding_value=0)
        translation = pad_sequence([torch.tensor(item["translation"], dtype=torch.long) for item in batch],
                                    batch_first=True, padding_value=0)
        return {
            "reference": reference.to(device),
            "translation": translation.to(device)
        }

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=text_detox_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=text_detox_collate_fn)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    
    model = create_model(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the PAD token
    optimizer = optim.Adam(model.parameters())

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the PAD token
    optimizer = torch.optim.Adam(model.parameters())

    # Train model
    for epoch in range(args.epochs):
        train_loss = train_detox_model(model, train_dataloader, optimizer, criterion, args.clip)
        eval_loss = evaluate_detox_model(model, eval_dataloader, criterion)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Eval Loss: {eval_loss}')
        
    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model for detoxifying text.")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train the model")
    parser.add_argument('--clip', type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    args = parser.parse_args()

    # Run the main function
    main(args)