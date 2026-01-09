import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import fasttext
import pickle
import os
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train_fixed_p1.txt")
VALID_PATH = os.path.join(BASE_DIR, "valid_fixed_p1.txt")
TEST_PATH = os.path.join(BASE_DIR, "test_fixed_p1.txt")
FASTTEXT_PATH = os.path.join(BASE_DIR, "../cc.id.300.bin")
EMBEDDING_DIM = 300
BATCH_SIZE = 32
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# 1. Data Loading Helper
def load_conll_from_file(file_path: str):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip().split("\n")

    data = []
    tokens = []
    tags = []

    for line in text:
        if line.strip() == "":
            if tokens:
                data.append((tokens, tags))
                tokens = []
                tags = []
        else:
            parts = line.split()
            if len(parts) >= 2:
                token, tag = parts[0], parts[-1]
                tokens.append(token)
                tags.append(tag)

    if tokens:
        data.append((tokens, tags))
    
    print(f"Loaded {len(data)} sentences.")
    return data

# 2. Model Definitions
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        return -self._log_likelihood(emissions, tags, mask)

    def _log_likelihood(self, emissions, tags, mask):
        batch_size, seq_length, num_tags = emissions.shape
        score = self._compute_score(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return (score - partition).sum()

    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_length = tags.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_length):
            mask_i = mask[:, i]
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, i - 1], tags[:, i]]
            score += (emit_score + trans_score) * mask_i

        last_tags = tags.gather(1, mask.sum(1).long().unsqueeze(1) - 1).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _compute_partition(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.shape
        alpha = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            emit_score = emissions[:, i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            next_alpha = alpha.unsqueeze(2) + emit_score + trans_score
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            mask_i = mask[:, i].unsqueeze(1)
            alpha = next_alpha * mask_i + alpha * (1 - mask_i)

        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.shape
        viterbi_score = self.start_transitions + emissions[:, 0]
        viterbi_path = []

        for i in range(1, seq_length):
            broadcast_score = viterbi_score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            viterbi_path.append(indices)
            mask_i = mask[:, i].unsqueeze(1)
            viterbi_score = next_score * mask_i + viterbi_score * (1 - mask_i)

        viterbi_score += self.end_transitions
        _, best_last_tag = viterbi_score.max(dim=1)

        best_paths = [best_last_tag]
        for indices in reversed(viterbi_path):
            best_last_tag = indices.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last_tag)

        best_paths.reverse()
        return torch.stack(best_paths, dim=1)

class BiGRUCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, embedding_matrix=None):
        super(BiGRUCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)

        self.bigru = nn.GRU(embedding_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, x, tags, lengths):
        mask = (x != 0).float()
        embeddings = self.embedding(x)
        embeddings = self.dropout(embeddings)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, _ = self.bigru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        gru_out = self.dropout(gru_out)
        emissions = self.hidden2tag(gru_out)
        loss = self.crf(emissions, tags, mask)
        return loss

    def predict(self, x, lengths):
        mask = (x != 0).float()
        embeddings = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, _ = self.bigru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        emissions = self.hidden2tag(gru_out)
        return self.crf.decode(emissions, mask)

class NERDataset(Dataset):
    def __init__(self, data, vocab, label_vocab):
        self.data = data
        self.vocab = vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        tag_ids = [self.label_vocab[tag] for tag in tags]
        return torch.LongTensor(token_ids), torch.LongTensor(tag_ids)

def collate_fn(batch):
    tokens, tags = zip(*batch)
    lengths = torch.LongTensor([len(t) for t in tokens])
    max_len = lengths.max().item()
    padded_tokens = torch.zeros(len(tokens), max_len, dtype=torch.long)
    padded_tags = torch.zeros(len(tags), max_len, dtype=torch.long)

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        padded_tokens[i, :len(token)] = token
        padded_tags[i, :len(tag)] = tag

    return padded_tokens, padded_tags, lengths

def build_vocab(data):
    word_counter = Counter()
    for tokens, _ in data:
        word_counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counter.most_common():
        vocab[word] = len(vocab)
    return vocab

def build_label_vocab(data):
    labels = set()
    for _, tags in data:
        labels.update(tags)
    return {label: idx for idx, label in enumerate(sorted(labels))}

def create_embedding_matrix(vocab, fasttext_model, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in ["<PAD>", "<UNK>"]:
            continue
        try:
            embedding_matrix[idx] = fasttext_model[word]
        except KeyError:
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
    return torch.FloatTensor(embedding_matrix)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for tokens, tags, lengths in tqdm(loader, desc="Training"):
        tokens, tags = tokens.to(device), tags.to(device)
        optimizer.zero_grad()
        loss = model(tokens, tags, lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, idx_to_label):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tokens, tags, lengths in tqdm(loader, desc="Evaluating"):
            tokens = tokens.to(device)
            predictions = model.predict(tokens, lengths)
            for i, length in enumerate(lengths):
                pred = predictions[i, :length].cpu().numpy()
                true = tags[i, :length].numpy()
                all_preds.extend([idx_to_label[p] for p in pred])
                all_labels.extend([idx_to_label[t] for t in true])
    return all_preds, all_labels

def main():
    # Load Data
    train_data = load_conll_from_file(TRAIN_PATH)
    valid_data = load_conll_from_file(VALID_PATH)
    test_data = load_conll_from_file(TEST_PATH)

    # Build Vocab
    vocab = build_vocab(train_data)
    label_vocab = build_label_vocab(train_data)
    idx_to_label = {v: k for k, v in label_vocab.items()}
    
    # Save Vocabs
    print("Saving vocabularies...")
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('label_vocab.pkl', 'wb') as f:
        pickle.dump(label_vocab, f)
    with open('idx_to_label.pkl', 'wb') as f:
        pickle.dump(idx_to_label, f)

    # Load FastText
    print("Loading FastText model...")
    fasttext_model = fasttext.load_model(FASTTEXT_PATH)
    embedding_matrix = create_embedding_matrix(vocab, fasttext_model, EMBEDDING_DIM)

    # Prepare Datasets
    train_dataset = NERDataset(train_data, vocab, label_vocab)
    valid_dataset = NERDataset(valid_data, vocab, label_vocab)
    test_dataset = NERDataset(test_data, vocab, label_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    model = BiGRUCRF(len(vocab), EMBEDDING_DIM, 256, len(label_vocab), embedding_matrix).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Training Loop
    best_f1 = 0
    patience_counter = 0
    patience = 3

    print("\nStarting Training...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_preds, val_labels = evaluate(model, valid_loader, DEVICE, idx_to_label)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        scheduler.step(train_loss)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("✓ Model saved!")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Final Evaluation
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_preds, test_labels = evaluate(model, test_loader, DEVICE, idx_to_label)
    
    unique_labels = sorted(set(test_labels))
    entity_labels = [l for l in unique_labels if l != 'O']
    print("\nTest Set Results:")
    print(classification_report(test_labels, test_preds, labels=entity_labels, digits=4))

if __name__ == "__main__":
    main()
