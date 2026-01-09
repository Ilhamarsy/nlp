import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import fasttext
import re
import os

# --- Model Definitions (Must match training code exactly) ---

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

        self.bigru = nn.GRU(
            embedding_dim,
            hidden_dim // 2, 
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

st.set_page_config(page_title="Named Entity Recognition", layout="wide")
st.title("Named Entity Recognition")
st.markdown("Extract entities from text using BiGRU-CRF + FastText.")

@st.cache_resource
def load_resources():
    with open('NER/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('NER/label_vocab.pkl', 'rb') as f:
        label_vocab = pickle.load(f)
    with open('NER/idx_to_label.pkl', 'rb') as f:
        idx_to_label = pickle.load(f)
        
    # Check if a trained model exists
    if os.path.exists('NER/best_model.pt'):
        # If trained model exists, we don't need FastText
        print("Trained model found. Skipping FastText loading.")
        return vocab, label_vocab, idx_to_label, None
    
    # Only load FastText if we don't have a trained model (fallback)
    fasttext_path = "../cc.id.300.bin"
    if not os.path.exists(fasttext_path):
        fasttext_path = "cc.id.300.bin" 
        
    if not os.path.exists(fasttext_path):
         st.error(f"FastText model not found at {fasttext_path}. Please check the path.")
         return None, None, None, None, None

    ft_model = fasttext.load_model(fasttext_path)
    
    return vocab, label_vocab, idx_to_label, ft_model

@st.cache_resource
def create_model(_vocab, _label_vocab, _ft_model, embedding_dim=300, hidden_dim=256):
    embedding_tensor = None
    
    # Only build embedding matrix if we have FastText loaded
    if _ft_model is not None:
        embedding_matrix = np.zeros((len(_vocab), embedding_dim))
        for word, idx in _vocab.items():
            if word in ["<PAD>", "<UNK>"]:
                continue
            try:
                embedding_matrix[idx] = _ft_model[word]
            except KeyError:
                embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
        embedding_tensor = torch.FloatTensor(embedding_matrix)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiGRUCRF(
        vocab_size=len(_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_tags=len(_label_vocab),
        embedding_matrix=embedding_tensor
    )
    
    if os.path.exists('NER/best_model.pt'):
        checkpoint = torch.load('NER/best_model.pt', map_location=device)
        model.load_state_dict(checkpoint)
    else:
        st.warning("NER/best_model.pt not found. Using untrained model.")
        
    model.to(device)
    model.eval()
    
    return model, device

with st.spinner("Loading models... This might take a minute initially."):
    vocab, label_vocab, idx_to_label, ft_model = load_resources()
    if vocab:
        model, device = create_model(vocab, label_vocab, ft_model)
    else:
        st.stop()

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"([.,!?():;])", r" \1 ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    tokens = sentence.split()
    return tokens

def predict(sentence):
    tokens = preprocess_sentence(sentence)
    
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    token_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    length_tensor = torch.LongTensor([len(tokens)])
    
    with torch.no_grad():
        predictions = model.predict(token_tensor, length_tensor)
        
    predicted_tag_ids = predictions.squeeze(0).cpu().numpy()
    predicted_labels = [idx_to_label[tag_id] for tag_id in predicted_tag_ids]
    
    return tokens, predicted_labels

def get_entity_color(tag):
    if "PPL" in tag: return "#ffadad"
    if "PLC" in tag: return "#caffbf"
    if "EVT" in tag: return "#ffd6a5"
    if "IND" in tag: return "#9bf6ff"
    if "FNB" in tag: return "#bdb2ff"
    return None

def visualize_result(tokens, tags):
    html_output = "<div style='line-height: 2.5;'>"
    
    for token, tag in zip(tokens, tags):
        color = get_entity_color(tag)
        if tag != "O" and color:
            display_tag = tag.split('-')[-1]
            html_output += f"<span style='background-color: {color}; padding: 4px 6px; border-radius: 4px; margin-right: 4px;'>{token} <small style='font-size: 0.6em; opacity: 0.7;'>{display_tag}</small></span> "
        else:
            html_output += f"{token} "
            
    html_output += "</div>"
    return html_output

user_input = st.text_area("Enter text:", height=100, placeholder="Contoh: Joko Widodo bertemu dengan Menteri Keuangan Sri Mulyani di Jakarta.")

if st.button("Analyze Entity"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            tokens, tags = predict(user_input)
            
            st.subheader("Result")
            st.markdown(visualize_result(tokens, tags), unsafe_allow_html=True)
            
            with st.expander("Detailed Token View"):
                st.table({"Token": tokens, "Tag": tags})
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.caption("NER Legend: PPL (Person), PLC (Place), EVT (Event), IND (Product), FNB (Food/Beverage)")