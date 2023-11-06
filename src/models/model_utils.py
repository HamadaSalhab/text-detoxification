import torch
import torch.nn as nn
from tqdm.auto import tqdm
import random
import os, sys
import pickle


class TextDetoxEncoder(nn.Module):
    """
    Encoder module for a sequence-to-sequence neural network used for text detoxification.

    Args:
        input_dim (int): Size of the input vocabulary.
        emb_dim (int): Size of the embedding layer.
        hid_dim (int): Size of the hidden states.
        n_layers (int): Number of layers in the LSTM.
        dropout (float): Dropout probability for regularization.

    Attributes:
        embedding (nn.Embedding): Embedding layer.
        rnn (nn.LSTM): LSTM layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Forward pass for the encoder.

        Args:
            src (torch.Tensor): Tensor of input sequences.

        Returns:
            tuple: Hidden and cell states to initialize the decoder.
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class TextDetoxDecoder(nn.Module):
    """
    Decoder module for a sequence-to-sequence neural network used for text detoxification.

    Args:
        output_dim (int): Size of the output vocabulary.
        emb_dim (int): Size of the embedding layer.
        hid_dim (int): Size of the hidden states.
        n_layers (int): Number of layers in the LSTM.
        dropout (float): Dropout probability for regularization.

    Attributes:
        output_dim (int): Size of the output vocabulary.
        embedding (nn.Embedding): Embedding layer.
        rnn (nn.LSTM): LSTM layer.
        fc_out (nn.Linear): Linear layer to predict next token.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        """
        Forward pass for the decoder.

        Args:
            input (torch.Tensor): Tensor of input tokens.
            hidden (torch.Tensor): Hidden state from the encoder or previous time step.
            cell (torch.Tensor): Cell state from the encoder or previous time step.

        Returns:
            tuple: Predicted token probabilities, new hidden state, new cell state.
        """
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class TextDetoxSeq2SeqModel(nn.Module):
    """
    Sequence-to-sequence model for text detoxification that uses the Encoder and Decoder.

    Args:
        encoder (TextDetoxEncoder): Encoder model.
        decoder (TextDetoxDecoder): Decoder model.
        device (torch.device): Device where the tensors will be stored. E.g., 'cpu' or 'cuda'.

    Attributes:
        encoder (TextDetoxEncoder): Encoder model.
        decoder (TextDetoxDecoder): Decoder model.
        device (torch.device): Device for tensors.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass for the entire sequence-to-sequence model.

        Args:
            src (torch.Tensor): Tensor of source sequences.
            trg (torch.Tensor): Tensor of target sequences.
            teacher_forcing_ratio (float): Probability to use the true target as the next input.

        Returns:
            torch.Tensor: Tensor of output probabilities for the target sequence.
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        
        input = trg[:, 0]  # Start token
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1) 
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        
        return outputs

def train_detox_model(model, iterator, optimizer, criterion, clip):
    """
    Function to train the detox model for one epoch.

    Args:
        model (nn.Module): The sequence-to-sequence detoxification model.
        iterator (DataLoader): A DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        criterion (nn.Module): The loss function.
        clip (float): The maximum value for gradient clipping.

    Returns:
        float: The average loss over the training data for this epoch.
    """
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(iterator, desc='Training', leave=False)
    for _, batch in enumerate(progress_bar):
        src = batch['reference']
        trg = batch['translation']

        optimizer.zero_grad()
        output = model(src, trg)

        # Flatten the output and target to pass into criterion
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)

def evaluate_detox_model(model, iterator, criterion):
    """
    Function to evaluate the detox model.

    Args:
        model (nn.Module): The sequence-to-sequence detoxification model.
        iterator (DataLoader): A DataLoader containing the evaluation data.
        criterion (nn.Module): The loss function.

    Returns:
        float: The average loss over the evaluation data.
    """
    model.eval()
    epoch_loss = 0

    progress_bar = tqdm(iterator, desc='Evaluating', leave=False)
    with torch.no_grad():
        for _, batch in enumerate(progress_bar):
            src = batch['reference']
            trg = batch['translation']

            output = model(src, trg, 0)  # Turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)

def create_model(device, checkpoint_path='./models/SEQ2SEQ_LSTMs.pth'):
    # Assume all hyperparameters and class definitions are available here
    # Create the instances of your Encoder, Decoder, and Seq2Seq model
    
    # Load the reference vocabulary from the pickle file
    VOCABS_DIR = './data/interim/'
    REFERENCE_VOCAB_PATH = VOCABS_DIR + 'reference_vocab.pkl'
    TRANSLATION_VOCAB_PATH = VOCABS_DIR + 'translation_vocab.pkl'
    
    with open(REFERENCE_VOCAB_PATH, 'rb') as f:
        reference_vocab = pickle.load(f)
    
    # Load the translation vocabulary from the pickle file
    with open(TRANSLATION_VOCAB_PATH, 'rb') as f:
        translation_vocab = pickle.load(f)
    
    
    INPUT_DIM = len(reference_vocab)
    OUTPUT_DIM = len(translation_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    # Load the model state from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = TextDetoxEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = TextDetoxDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = TextDetoxSeq2SeqModel(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model