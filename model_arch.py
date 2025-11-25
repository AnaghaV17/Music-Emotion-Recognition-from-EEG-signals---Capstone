
import torch
import torch.nn as nn
import numpy as np


# === Model Architectures provided by the user ===
class GESSNetWithOpenSMILE(nn.Module):
    def __init__(self, num_classes, input_channels=64, dropout_rate=0.3, song_feature_dim=None):
        super().__init__()

        # EEG processing branch
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=0, bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(input_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate)
        )

        self.additional_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 16))
        )

        # EEG feature extraction
        self.fc_eeg = nn.Sequential(
            nn.Linear(64 * 1 * 16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)
        )

        # Song feature processing with attention
        self.song_attention = nn.Sequential(
            nn.Linear(song_feature_dim, song_feature_dim // 4),
            nn.ReLU(),
            nn.Linear(song_feature_dim // 4, song_feature_dim),
            nn.Sigmoid()
        )

        self.song_project = nn.Sequential(
            nn.Linear(song_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)
        )

        # Cross-attention between EEG and song features
        self.cross_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout_rate, batch_first=True)

        # Final classifier
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg, song_feat):
        # EEG processing
        x = eeg.unsqueeze(1)  # (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.additional_conv(x)
        x = x.flatten(start_dim=1)
        eeg_features = self.fc_eeg(x)

        # Song feature processing with attention
        song_attention_weights = self.song_attention(song_feat)
        song_feat_attended = song_feat * song_attention_weights
        song_features = self.song_project(song_feat_attended)

        # Cross-attention
        eeg_features_expanded = eeg_features.unsqueeze(1)
        song_features_expanded = song_features.unsqueeze(1)

        attended_eeg, _ = self.cross_attention(eeg_features_expanded, song_features_expanded, song_features_expanded)
        attended_eeg = attended_eeg.squeeze(1)

        # Combine features
        combined = torch.cat([attended_eeg, song_features], dim=1)
        return self.fc_combined(combined)


class GESSNetWithYAMNet(nn.Module):
    def __init__(self, num_classes, input_channels=64, dropout_rate=0.3, song_feature_dim=None):
        super().__init__()

        # EEG processing branch
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=0, bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(input_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate)
        )

        self.additional_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 16))
        )

        # EEG feature extraction
        self.fc_eeg = nn.Sequential(
            nn.Linear(64 * 1 * 16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)
        )

        # Song feature processing with attention
        self.song_attention = nn.Sequential(
            nn.Linear(song_feature_dim, song_feature_dim // 4),
            nn.ReLU(),
            nn.Linear(song_feature_dim // 4, song_feature_dim),
            nn.Sigmoid()
        )

        self.song_project = nn.Sequential(
            nn.Linear(song_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)
        )

        # Cross-attention between EEG and song features
        self.cross_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=dropout_rate, batch_first=True)

        # Final classifier
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg, song_feat):
        # EEG processing
        x = eeg.unsqueeze(1)  # (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.additional_conv(x)
        x = x.flatten(start_dim=1)
        eeg_features = self.fc_eeg(x)

        # Song feature processing with attention
        song_attention_weights = self.song_attention(song_feat)
        song_feat_attended = song_feat * song_attention_weights
        song_features = self.song_project(song_feat_attended)

        # Cross-attention
        eeg_features_expanded = eeg_features.unsqueeze(1)
        song_features_expanded = song_features.unsqueeze(1)

        attended_eeg, _ = self.cross_attention(eeg_features_expanded, song_features_expanded, song_features_expanded)
        attended_eeg = attended_eeg.squeeze(1)

        # Combine features
        combined = torch.cat([attended_eeg, song_features], dim=1)
        return self.fc_combined(combined)


def get_model_opensmile(name=None, song_feature_dim=None, num_classes=None, input_channels=64):
    # Provide defaults if not inferred; caller should attempt to infer from state_dict
    if song_feature_dim is None:
        song_feature_dim = 128
    if num_classes is None:
        num_classes = 4
    if input_channels is None:
        input_channels = 64
    return GESSNetWithOpenSMILE(num_classes=num_classes, input_channels=input_channels, dropout_rate=0.3, song_feature_dim=song_feature_dim)


def get_model_yamnet(name=None, song_feature_dim=None, num_classes=None, input_channels=64):
    if song_feature_dim is None:
        song_feature_dim = 128
    if num_classes is None:
        num_classes = 4
    if input_channels is None:
        input_channels = 64
    return GESSNetWithYAMNet(num_classes=num_classes, input_channels=input_channels, dropout_rate=0.3, song_feature_dim=song_feature_dim)


def get_model(name=None, song_feature_dim=None, num_classes=None, input_channels=64):
    # generic fallback
    return get_model_opensmile(name, song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
