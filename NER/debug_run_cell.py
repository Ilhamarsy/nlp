import torch
import torch.nn as nn
import fasttext
import numpy as np
import pickle
import os

# Define classes first (copied from notebook)
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags

        # Transition scores: transitions[i][j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        """Compute negative log likelihood (for training)"""
        return -self._log_likelihood(emissions, tags, mask)

    def _log_likelihood(self, emissions, tags, mask):
        batch_size, seq_length, num_tags = emissions.shape

        # Compute score of the given tag sequence
        score = self._compute_score(emissions, tags, mask)

        # Compute partition function (normalization)
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
        """Viterbi decoding"""
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

        # Backtrack
        best_paths = [best_last_tag]
        for indices in reversed(viterbi_path):
            best_last_tag = indices.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last_tag)

        best_paths.reverse()
        return torch.stack(best_paths, dim=1)

# BiGRU-CRF Model
class BiGRUCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, embedding_matrix=None):
        super(BiGRUCRF, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)

        self.bigru = nn.GRU(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 because bidirectional
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, x, tags, lengths):
        mask = (x != 0).float()

        embeddings = self.embedding(x)
        embeddings = self.dropout(embeddings)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.bigru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        gru_out = self.dropout(gru_out)
        emissions = self.hidden2tag(gru_out)

        loss = self.crf(emissions, tags, mask)
        return loss

    def predict(self, x, lengths):
        mask = (x != 0).float()

        embeddings = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.bigru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        emissions = self.hidden2tag(gru_out)
        predictions = self.crf.decode(emissions, mask)

        return predictions

# --- Main run logic ---

print("Loading vocab...")
try:
    with open('vocab.pkl', 'rb') as f:
        loaded_vocab = pickle.load(f)
    print(f"Loaded vocab size: {len(loaded_vocab)}")

    with open('label_vocab.pkl', 'rb') as f:
        loaded_label_vocab = pickle.load(f)
    print(f"Loaded label_vocab size: {len(loaded_label_vocab)}")
except FileNotFoundError:
    print("Error: vocab.pkl or label_vocab.pkl not found!")
    exit(1)

print("Loading fasttext model...")
fasttext_path = "../cc.id.300.bin"
if not os.path.exists(fasttext_path):
    print(f"Error: Fasttext model not found at {fasttext_path}")
    exit(1)
fasttext_model = fasttext.load_model(fasttext_path)
EMBEDDING_DIM = 300

# Create embedding matrix
def create_embedding_matrix(vocab, fasttext_model, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in ["<PAD>", "<UNK>"]:
            continue
        try:
            embedding_matrix[idx] = fasttext_model[word]
        except KeyError:
            # Random initialization for OOV words
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)

    # FIX: Use torch.tensor for safety
    return torch.tensor(embedding_matrix).float()

print("Creating embedding matrix...")
loaded_embedding_matrix = create_embedding_matrix(loaded_vocab, fasttext_model, EMBEDDING_DIM)

print("Initializing model...")
# Re-initialize the model architecture using loaded vocab and label vocab
model = BiGRUCRF(
    vocab_size=len(loaded_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=256,
    num_tags=len(loaded_label_vocab),
    embedding_matrix=loaded_embedding_matrix
)
# .to(device)

print("Loading state dict...")
if os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt'))
    print("Model loaded successfully from best_model.pt using loaded vocab and label vocab.")
else:
    print("Error: best_model.pt not found!")
