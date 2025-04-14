import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=23, d_model=512, nhead=8, num_layers=8, num_classes=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Add character-level CNN
        self.char_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            d_model, d_model // 2, 
            num_layers=2, 
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # CNN processing
        x_conv = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x_conv = self.char_conv(x_conv)
        x = x_conv.transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        # Transformer processing
        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer(x)
        
        # LSTM processing
        x, _ = self.lstm(x)  # Now x shape is (batch_size, seq_len, d_model)
        
        # Classification
        x = self.classifier(x)  # (batch_size, seq_len, num_classes)
        return x  # Return complete sequence prediction results 