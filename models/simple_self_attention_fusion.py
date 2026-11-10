import torch
import torch.nn as nn


class SimpleSelfAttentionFusion(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_layers=2,
        hidden_size=768,
        img_hidden=512,
        mlp_dim=1536,
        dropout_rate=0.5,
        num_heads=12,
        attention_dropout_rate=0.3,
        token_mode="all",
    ):
        super().__init__()
        if token_mode not in ("cls", "all"):
            raise ValueError("token_mode must be 'cls' or 'all', got {}".format(token_mode))
        self.hidden_size = hidden_size
        self.token_mode = token_mode

        self.img_proj = nn.Linear(img_hidden, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cls_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.img_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        encoder_dropout = attention_dropout_rate if attention_dropout_rate is not None else dropout_rate

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=encoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        self.head = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_type_embed, std=0.02)
        nn.init.trunc_normal_(self.img_type_embed, std=0.02)
        nn.init.trunc_normal_(self.text_type_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _select_img_tokens(self, img_features):
        if img_features.dim() == 2:
            return img_features.unsqueeze(1)
        if img_features.dim() == 3:
            if self.token_mode == "cls":
                return img_features[:, :1, :]
            return img_features
        if img_features.dim() == 4:
            if self.token_mode == "cls":
                return img_features[:, :, 0, :]
            batch_size, num_moda, num_tokens, dim = img_features.shape
            return img_features.reshape(batch_size, num_moda * num_tokens, dim)
        raise ValueError(
            "img_features must have shape [B, D], [B, T, D], or [B, M, T, D], got {}".format(
                tuple(img_features.shape)
            )
        )

    def _select_text_tokens(self, text_features):
        if text_features is None:
            return None
        if text_features.dim() == 2:
            return text_features.unsqueeze(1)
        if text_features.dim() == 3:
            if self.token_mode == "cls":
                return text_features[:, :1, :]
            return text_features
        raise ValueError(
            "text_features must have shape [B, D] or [B, T, D], got {}".format(
                tuple(text_features.shape)
            )
        )

    def forward(self, img_features, text_features=None):
        img_features = self._select_img_tokens(img_features)
        text_features = self._select_text_tokens(text_features)

        img_tokens = self.img_proj(img_features) + self.img_type_embed
        tokens = [img_tokens]

        if text_features is not None:
            text_tokens = self.text_proj(text_features) + self.text_type_embed
            tokens.append(text_tokens)

        batch_size = img_tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1) + self.cls_type_embed
        x = torch.cat([cls_token] + tokens, dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits
