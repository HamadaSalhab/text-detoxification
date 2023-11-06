import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from model_utils import TextDetoxEncoder, TextDetoxDecoder, TextDetoxSeq2SeqModel, create_model  # Ensure these are defined or imported appropriately

# Path to the saved model and vocab files
MODEL_PATH = './models/SEQ2SEQ_LSTMs.pth'  # Update with the path to your saved model
VOCABS_DIR = './data/interim/'
REFERENCE_VOCAB_PATH = VOCABS_DIR + 'reference_vocab.pkl'
TRANSLATION_VOCAB_PATH = VOCABS_DIR + 'translation_vocab.pkl'

def load_vocabs():
    with open(REFERENCE_VOCAB_PATH, 'rb') as f:
        reference_vocab = pickle.load(f)
    with open(TRANSLATION_VOCAB_PATH, 'rb') as f:
        translation_vocab = pickle.load(f)
    return reference_vocab, translation_vocab

    
# Define special tokens that will be used in the data preprocessing
PAD_TOKEN = '<pad>'  # Token used for padding sentences to the same length
SOS_TOKEN = '<sos>'  # Start-of-sentence token
EOS_TOKEN = '<eos>'  # End-of-sentence token

def detoxify_sentence(sentence, reference_vocab, translation_vocab, model, device, max_len=50):
    model.eval()
    
    # Tokenize the sentence, add the <sos> and <eos> tokens, and numericalize
    tokens = [reference_vocab.get(token, reference_vocab[PAD_TOKEN]) for token in sentence.split()]
    numericalized_tokens = [reference_vocab[SOS_TOKEN]] + tokens + [reference_vocab[EOS_TOKEN]]
    
    # Convert to Tensor and add a batch dimension
    src_tensor = torch.LongTensor(numericalized_tokens).unsqueeze(0).to(device)
    
    # Predict the target sequence
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        trg_indexes = [translation_vocab[SOS_TOKEN]]

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
            # Get the predicted next token (the one with the highest probability)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            # If the <eos> token is predicted, stop
            if pred_token == translation_vocab[EOS_TOKEN]:
                break
    
    # Convert the predicted numerical tokens to words
    trg_tokens = [list(translation_vocab.keys())[list(translation_vocab.values()).index(idx)] for idx in trg_indexes]
    
    # Return the words after the <sos> token
    return trg_tokens[1:-1]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabularies and model
    reference_vocab, translation_vocab = load_vocabs()
    model = create_model(device)

    # Prompt the user for a sentence to detoxify
    print("Please enter a sentence to detoxify:")
    user_sentence = input()

    # Perform the prediction
    detoxified_tokens = detoxify_sentence(user_sentence, reference_vocab, translation_vocab, model, device)
    print("Detoxified sentence:")
    print(" ".join(detoxified_tokens))
