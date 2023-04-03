import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResAtt(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MultiResAtt, self).__init__()
        num_IMUs = 1
        self.initial_block = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.residual_modules = nn.ModuleList([nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        ) for _ in range(num_IMUs)])

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.bigru = nn.GRU(input_size=256 * num_IMUs, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.initial_block(x)
        
        # Apply residual modules
        residual_outputs = []
        for residual_module in self.residual_modules:
            residual_outputs.append(residual_module(x))

        # Concatenate the outputs of residual modules
        x = torch.cat(residual_outputs, dim=1)
        
        x = self.adaptive_avg_pool(x).squeeze(dim=-1)

        # Add an extra dimension for the GRU input
        x = x.unsqueeze(1)
        
        gru_out, _ = self.bigru(x)
        attention_weights = self.attention(gru_out).squeeze(dim=-1)
        attention_weights = torch.softmax(attention_weights, dim=-1).unsqueeze(-1)
        attention_out = torch.bmm(gru_out.permute(0, 2, 1), attention_weights).squeeze(dim=-1)

        output = self.fc(attention_out)
        return output