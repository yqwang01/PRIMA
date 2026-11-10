import torch
from torch import nn

class FusionMLP(nn.Module):
    """
    Unified fusion MLP for two modes:
      - mode='full':
          x_img: [B, M, 512] -> flatten -> [B, M*512]
          x_txt: [B, 768]    -> Linear(768->512) -> [B, 512]
          concat -> MLP -> logits(2)
      - mode='segments':
          x_img: [B, M, 512] -> flatten -> [B, M*512]
          x_txt: [B, K, 768] -> Linear(768->512) per segment -> [B, K, 512]
                 flatten -> [B, K*512]
          concat -> MLP -> logits(2)
    """
    def __init__(self,
                 num_moda: int,
                 mode: str = "full",          # "full" or "segments"
                 text_max_segs: int = 16,     # used only when mode='segments'
                 img_dim: int = 512,
                 txt_dim: int = 768,
                 hidden1: int = 512,
                 hidden2: int = 128,
                 num_classes: int = 2,
                 dropout_rate: float = 0.1,
                 ):
        super().__init__()
        assert mode in ("full", "segments")
        self.mode = mode
        self.text_max_segs = text_max_segs
        self.img_dim = img_dim
        self.txt_dim = txt_dim

        # project text to the same dim as image feature (512)
        self.txt_proj = nn.Linear(txt_dim, img_dim)

        # input dimension to the MLP
        if self.mode == "full":
            in_dim = num_moda * img_dim + img_dim          # M*512 + 512
        else:  # segments
            in_dim = num_moda * img_dim + text_max_segs * img_dim  # M*512 + K*512

        # a small MLP head
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.relu1 = nn.ReLU(True)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU(True)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x_img, x_txt):
        """
        Args:
          x_img: [B, M, 512]
          x_txt:
            - mode='full'     -> [B, M, 768]
            - mode='segments' -> [B, K, 768]  (K must equal text_max_segs used in Dataset)
        """
        B = x_img.size(0)
        x_img_cls = x_img[:, :, 0, :]   # [Batch, Dim]
        x_img_flat = x_img_cls.reshape(B, -1)  # [B, M*512]
        x_txt_cls = x_txt[:, 0, :]   # [Batch, Dim]

        if self.mode == "full":
            # [B, 768] -> [B, 512]
            x_txt_proj = self.txt_proj(x_txt_cls)  # [B, 512]
            x = torch.cat([x_img_flat, x_txt_proj], dim=1)
        else:
            # [B, K, 768] -> [B, K, 512] -> [B, K*512]
            # (assume K == self.text_max_segs by Dataset padding/truncation)
            x_txt_proj = self.txt_proj(x_txt)          # [B, K, 512]
            x_txt_flat = x_txt_proj.reshape(B, -1)     # [B, K*512]
            x = torch.cat([x_img_flat, x_txt_flat], dim=1)

        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x
