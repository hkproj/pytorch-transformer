from model import build_transformer, causal_mask
import spacy
import os
from os.path import exists
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from pathlib import Path

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        # Index = 0 means to extract the first sentence from the pair (German)
        yield_tokens(train + val + test,
                     lambda text: tokenize(text, spacy_de), index=0),
        min_freq=2,  # Minimum frquency to include a token in the vocabulary
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        # Index = 1 means to extract the second sentence from the pair (English)
        yield_tokens(train + val + test,
                     lambda text: tokenize(text, spacy_en), index=1),
        min_freq=2,  # Minimum frquency to include a token in the vocabulary
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    assert vocab_src['<s>'] == vocab_tgt['<s>']
    assert vocab_src['</s>'] == vocab_tgt['</s>']
    assert vocab_src['<blank>'] == vocab_tgt['<blank>']
    assert vocab_src['<unk>'] == vocab_tgt['<unk>']

    # Set the position of the <unk> token to be used for OOV words.
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    return vocab_src, vocab_tgt


class GermanEnglishDataset(Dataset):

    def __init__(self, spacy_de, spacy_en, vocab_src, vocab_tgt, tokenize_src, tokenize_tgt, seq_len, split, device):
        super().__init__()
        self.split = split
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.spacy_de = spacy_de
        self.spacy_en = spacy_en
        self.seq_len = seq_len
        self.device = device
        self.tokenize_src = tokenize_src
        self.tokenize_tgt = tokenize_tgt

        self.sos_token = torch.tensor([self.vocab_tgt["<s>"]], device=device)
        self.eos_token = torch.tensor([self.vocab_tgt["</s>"]], device=device)
        self.pad_token = torch.tensor(
            [self.vocab_tgt["<blank>"]], device=device)

        ds = datasets.Multi30k(language_pair=("de", "en"), split=split)

        self.data = []
        for entry in ds:
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_target_pair = self.data[idx]
        src_text = src_target_pair[0]
        tgt_text = src_target_pair[1]

        # Transform the text into tokens
        enc_input_tokens = self.vocab_src(self.tokenize_src(src_text))
        dec_input_tokens = self.vocab_tgt(self.tokenize_tgt(tgt_text))

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64, device=self.device),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64, device=self.device),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64,device=self.device),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64, device=self.device),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64, device=self.device),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64, device=self.device),
            ],
            dim=0,
        )

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def greedy_decode(model, source, source_mask, vocab_src, vocab_tgt, max_len, device):
    sos_idx = vocab_src['<s>']
    eos_idx = vocab_src['</s>']

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(
            1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask,
                           decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, vocab_src, vocab_tgt, max_len, device, print_msg):
    model.eval()
    num_examples = 2
    count = 0

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, vocab_src, vocab_tgt, max_len, device)

            source_text = batch["src_text"]
            target_text = batch["tgt_text"]
            model_out_text = " ".join([vocab_tgt.vocab.itos_[idx] for idx in model_out.cpu().numpy()])

            # get the console window width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'{bcolors.OKBLUE}German:{bcolors.ENDC} ':>20}{source_text}")
            print_msg(f"{f'{bcolors.OKCYAN}English:{bcolors.ENDC} ':>20}{target_text}")
            print_msg(f"{f'{bcolors.OKGREEN}Model:{bcolors.ENDC} ':>20}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "accum_iter": 10,
        "base_lr": 1.0,
        "seq_len": 72,
        "warmup": 1000,
        "d_model": 512
    }

def get_ds(config):
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    train_ds = GermanEnglishDataset(spacy_de, spacy_en, vocab_src, vocab_tgt, lambda text: tokenize(
        text, spacy_de), lambda text: tokenize(text, spacy_en), 72, 'train', 'cpu')

    # Split the dataset into train and validation
    train_ds_size = int(0.9 * len(train_ds))
    val_ds_size = len(train_ds) - train_ds_size
    train_ds, val_ds = random_split(train_ds, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, vocab_src, vocab_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    train_dataloader, val_dataloader, vocab_src, vocab_tgt = get_ds(config)
    model = get_model(config, len(vocab_src), len(vocab_tgt)).to(device)

    # Optimizer with its LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, config['d_model'], factor=1, warmup=config["warmup"]
        ),
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab_tgt["<blank>"], label_smoothing=0.1).to(device)

    global_step = 0
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_step = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Compare the output with the label
            label = batch['label'].to(device)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, len(vocab_tgt)), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{lr_scheduler.get_last_lr()[0]:6.1e}", "epoch step": f"{epoch_step:03d}"})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()  # Update learning rate schedule

            global_step += 1
            epoch_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, vocab_src, vocab_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

        model_folder = "weights"
        model_filename = f"tmodel_{epoch:02d}.pt"
        if epoch == config['num_epochs'] - 1:
            model_filename = "tmodel_final.pt"
        # Save the model
        torch.save(model.state_dict(), str(Path('.') / model_folder / model_filename))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model()
