import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class TextDetoxDataset(Dataset):
    """
    A PyTorch Dataset class that loads numericalized text data for text detoxification.

    The dataset class is responsible for loading and converting the stringified numericalized text data 
    from a pandas DataFrame into PyTorch tensors that can be processed by a neural network model.

    Attributes:
        reference_numericalized (pandas.Series): A series of lists representing the numericalized
                                                 tokens of the reference sentences.
        translation_numericalized (pandas.Series): A series of lists representing the numericalized
                                                   tokens of the translation sentences.
    """

    def __init__(self, dataframe):
        """
        Initializes the dataset with the given dataframe.

        Args:
            dataframe (pandas.DataFrame): A DataFrame containing the numericalized text data.
        """
        def string_to_list(s):
            """
            Helper function to convert a string representation of a list into an actual list of integers.
            If the string is not in a JSON format, it attempts to parse it manually.

            Args:
                s (str): A string representation of a list.

            Returns:
                list: A list of integers represented by the string.
            """
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return [int(item) for item in s.strip('[]').split(', ')]

        self.reference_numericalized = dataframe['reference_numericalized'].apply(string_to_list)
        self.translation_numericalized = dataframe['translation_numericalized'].apply(string_to_list)

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.reference_numericalized)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary with 'reference' and 'translation' keys containing the numericalized
                  tokens of the reference and translation sentences, respectively.
        """
        input_ids = self.reference_numericalized.iloc[idx]
        labels = self.translation_numericalized.iloc[idx]

        return {
            "reference": input_ids,
            "translation": labels
        }


class TextDetoxDataLoader(DataLoader):
    """
    A PyTorch DataLoader class customized for the text detoxification dataset.

    This data loader extends the PyTorch DataLoader and is responsible for batching the data and
    providing it to the model during training or evaluation.

    Inherits all methods and attributes from the DataLoader class.
    """

    def __init__(self, dataset, collate_fn, batch_size=8, shuffle=True):
        """
        Initializes the data loader with the given dataset and collate function.

        Args:
            dataset (Dataset): An instance of the TextDetoxDataset containing the data.
            collate_fn (callable): A function to merge a list of samples to form a mini-batch.
            batch_size (int, optional): The number of samples in each batch. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_dataloaders():
    

    df = get_dataframe()

    # Train/Test Split
    train, eval = train_test_split(df, test_size=0.2, shuffle=False)


    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    

    train_dataset = TextDetoxDataset(train)
    eval_dataset = TextDetoxDataset(eval)


    def text_detox_collate_fn(batch):
        reference = pad_sequence([torch.tensor(item["reference"], dtype=torch.long) for item in batch],
                                    batch_first=True, padding_value=0)
        translation = pad_sequence([torch.tensor(item["translation"], dtype=torch.long) for item in batch],
                                    batch_first=True, padding_value=0)
        return {
            "reference": reference.to(device),
            "translation": translation.to(device)
        }

    train_dataloader = TextDetoxDataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=text_detox_collate_fn)
    eval_dataloader = TextDetoxDataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=text_detox_collate_fn)

    return train_dataloader, eval_dataloader