"""
深度混合神经网络（DHNN）
关键修复：
1. 新增 input_proj，修复 Attention 残差连接的维度对齐问题
2. 分类器加深：192→128→64→7
3. 支持所有消融开关
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DHNN(nn.Module):
    """
    Deep Hybrid Neural Network for HSK Text Grading
    
    架构：
    - 嵌入层 (300维) → 输入投影到 384维
    - CNN (多尺度) → 768维
    - BiLSTM → 1152维
    - Attention → 1152维
    - 拼接: 3072维
    - 投影层: 3072 → 1920
    - 融合网络: 1920 → 768 → 384 → 192
    - 分类器: 192 → 128 → 64 → 7
    """
    
    def __init__(self,
                 vocab_size=60000,
                 embed_dim=300,
                 hidden_dim=384,
                 num_layers=3,
                 num_classes=7,
                 conv_channels=128,
                 attention_heads=8,
                 max_length=200,
                 dropout=0.4,
                 use_cnn=True,
                 use_lstm=True,
                 use_attention=True,
                 use_aux=True,
                 use_multi_scale=True,
                 use_label_smoothing=True):
        super().__init__()
        
        self.cfg = {
            'vocab_size': vocab_size, 'embed_dim': embed_dim,
            'hidden_dim': hidden_dim, 'num_layers': num_layers,
            'num_classes': num_classes, 'conv_channels': conv_channels,
            'attention_heads': attention_heads, 'max_length': max_length,
            'use_cnn': use_cnn, 'use_lstm': use_lstm,
            'use_attention': use_attention, 'use_aux': use_aux,
            'use_multi_scale': use_multi_scale,
            'use_label_smoothing': use_label_smoothing,
        }
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embed_dim))
        self.embed_dropout = nn.Dropout(0.1)
        
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        if use_cnn:
            kernel_sizes = [2, 3, 4] if use_multi_scale else [3]
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, conv_channels, k, padding=k // 2)
                for k in kernel_sizes
            ])
            self.conv_norms = nn.ModuleList([
                nn.BatchNorm1d(conv_channels) for _ in kernel_sizes
            ])
            self.cnn_out_dim = conv_channels * len(kernel_sizes) * 2
        else:
            self.cnn_out_dim = 0
        
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim // 2,
                num_layers=num_layers, batch_first=True,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            self.lstm_norm = nn.LayerNorm(hidden_dim)
            self.lstm_out_dim = hidden_dim * 3
        else:
            self.lstm_out_dim = 0
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim, attention_heads,
                dropout=0.1, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn_out_dim = hidden_dim * 3
        else:
            self.attn_out_dim = 0
        
        self.total_concat = self.cnn_out_dim + self.lstm_out_dim + self.attn_out_dim
        if self.total_concat == 0:
            self.total_concat = embed_dim * 2
        
        self.projection = nn.Linear(self.total_concat, 1920)
        
        self.fusion = nn.Sequential(
            nn.Linear(1920, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        self.main_cls = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        if use_aux:
            self.aux_cls = nn.Sequential(
                nn.Linear(192, 64),
                nn.GELU(),
                nn.Linear(64, num_classes)
            )
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] DHNN initialized with {n_params:,} parameters")
        print(f"[Model] CNN={self.cnn_out_dim}, LSTM={self.lstm_out_dim}, "
              f"Attn={self.attn_out_dim}, Total={self.total_concat}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if input_ids.size(1) <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :input_ids.size(1), :]
        x = self.embed_dropout(x)
        
        features = []
        
        if self.cfg['use_cnn']:
            x_t = x.transpose(1, 2)
            conv_feats = []
            for conv, norm in zip(self.convs, self.conv_norms):
                c = norm(conv(x_t))
                c = F.gelu(c)
                conv_feats.append(F.adaptive_max_pool1d(c, 1).squeeze(-1))
                conv_feats.append(F.adaptive_avg_pool1d(c, 1).squeeze(-1))
            features.append(torch.cat(conv_feats, dim=1))
        
        x_proj = self.input_proj(x)
        
        seq_input = x_proj
        if self.cfg['use_lstm']:
            lstm_out, _ = self.lstm(x_proj)
            lstm_out = self.lstm_norm(lstm_out)
            features.append(torch.max(lstm_out, dim=1)[0])
            features.append(torch.mean(lstm_out, dim=1))
            features.append(lstm_out[:, -1, :])
            seq_input = lstm_out
        
        if self.cfg['use_attention']:
            key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
            attn_out, _ = self.attention(
                seq_input, seq_input, seq_input,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            attn_out = self.attn_norm(seq_input + attn_out)
            features.append(torch.max(attn_out, dim=1)[0])
            features.append(torch.mean(attn_out, dim=1))
            features.append(attn_out[:, -1, :])
        
        if not features:
            features.append(torch.max(x, dim=1)[0])
            features.append(torch.mean(x, dim=1))
        
        combined = torch.cat(features, dim=1)
        projected = self.projection(combined)
        fused = self.fusion(projected)
        main_logits = self.main_cls(fused)
        
        aux_logits = None
        if self.cfg['use_aux']:
            aux_logits = self.aux_cls(fused)
        
        return main_logits, aux_logits


class FocalLoss(nn.Module):
    """Focal Loss with class weights and label smoothing"""
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, use_smoothing=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing if use_smoothing else 0.0
        self.use_smoothing = use_smoothing
    
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        focal_weight = (1 - pt) ** self.gamma
        
        if self.use_smoothing and self.smoothing > 0:
            smooth_labels = torch.full_like(
                logits, self.smoothing / (logits.size(-1) - 1)
            )
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        else:
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0)
        
        loss = -(focal_weight.unsqueeze(1) * log_probs * smooth_labels).sum(dim=1)
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


class CrossEntropyLossWrapper(nn.Module):
    """标准交叉熵损失（用于消融）"""
    def __init__(self, alpha=None, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        if self.alpha is not None:
            return F.cross_entropy(logits, targets, weight=self.alpha,
                                   label_smoothing=self.smoothing)
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing)