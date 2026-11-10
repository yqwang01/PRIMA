# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import torch.nn.functional as F

class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.0):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)

        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy() # [2N, *]
        weights = self.kernel(all_labels, all_labels) # type: ignore # [2N, 2N]
        weights = weights * (1 - np.eye(2*N)) # puts 0 on the diagonal
        weights /= weights.sum(axis=1)
        # if 'rbf' kernel and sigma->0, we retrieve the classical NTXenLoss (without labels)
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0) # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        loss = -1./N * (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum()

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss

    def __str__(self):
        return "{}(temp={}, kernel={}, sigma={})".format(type(self).__name__, self.temperature,
                                                         self.kernel.__name__, self.sigma) # type: ignore



class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j, labels=None, reduction='mean'):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs, reduction=reduction)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs, reduction=reduction)

        if self.return_logits:
            return loss_i, loss_j, sim_zij, correct_pairs

        return loss_i, loss_j

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, outputs, targets):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss


def selfpaced_weighted(loss, gamma, pace_type=None):
    l_i_j = loss.detach()
    if pace_type == "hard":
        weight = (l_i_j <= gamma).float()
    elif pace_type == "linear":
        weight = torch.max(1 - 1 / gamma * l_i_j, torch.zeros_like(l_i_j))
    else:
        weight = torch.ones_like(l_i_j)
    assert torch.logical_and(weight >= 0, weight <= 1).any()
    loss = torch.mean(loss * weight)

    return loss# , weight



class AdaSPLoss(nn.Module):
    """
    AdaSP loss for (B, N, D) input.
    - Each sample has N views (or attributes).
    - Views within same sample are positives.
    - Views across different samples are negatives.
    """

    def __init__(self, temp=0.04, loss_type='adasp'):
        super().__init__()
        self.temp = temp
        self.scale = 1. / temp
        self.loss_type = loss_type

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) tensor
        Returns:
            loss: scalar
        """
        B, N, D = feats.shape
        device = feats.device

        # Normalize features across last dimension
        feats = F.normalize(feats, dim=2)  # (B, N, D)

        # Flatten to (B*N, D)
        feats_flat = feats.view(B * N, D)

        # Compute full similarity matrix: sim_mat[i,j] = cos(feat_i, feat_j)
        sim_mat = torch.matmul(feats_flat, feats_flat.T) * self.scale  # (BN, BN)

        # ------------------------------
        # HARD-HARD (HH)
        # ------------------------------
        eye_B = torch.eye(B, device=device)
        block_mask = torch.kron(eye_B, torch.ones((N, N), device=device))  # (BN, BN)
        mask_HH = block_mask.clone()
        mask_HH[mask_HH == 1] = -1.0
        mask_HH[mask_HH == 0] = 1.0

        sim_HH = torch.exp(sim_mat * mask_HH)  # (BN, BN)

        right = torch.kron(eye_B, torch.ones((N, 1), device=device))   # (BN, B)
        left = torch.kron(eye_B, torch.ones((1, N), device=device))    # (B, BN)

        ID_HH = left @ (sim_HH @ right)  # (B, B)

        pos_diag = torch.diag(ID_HH)
        pos_diag[pos_diag == 0] = 1.0
        ID_HH[range(B), range(B)] = (1.0 / pos_diag).to(ID_HH.dtype)
        ID_HH_L1 = F.normalize(ID_HH, p=1, dim=1)

        # ------------------------------
        # HARD-EASY (HE) with correct mask
        # ------------------------------
        mask_HE = mask_HH.clone()
        sim_HE = torch.exp(sim_mat * mask_HE)

        ID_HE = sim_HE @ right  # (BN, B)

        sample_ids = torch.arange(B, device=device).repeat_interleave(N)  # shape: (BN,)
        row_ids = sample_ids.view(-1, 1)       # (BN, 1)
        col_ids = torch.arange(B, device=device).view(1, -1)  # (1, B)
        pos_mask_HE = (row_ids == col_ids)     # (BN, B)

        pos_HE = ID_HE[pos_mask_HE]
        pos_HE[pos_HE == 0] = 1.0
        ID_HE[pos_mask_HE] = (1.0 / pos_HE).to(ID_HE.dtype)

        ID_HE = left @ ID_HE  # (B, B)
        ID_HE_L1 = F.normalize(ID_HE, p=1, dim=1)

        # ------------------------------
        # Adaptive Fusion (AdaSP)
        # ------------------------------
        l_sim = torch.log(torch.diag(ID_HH))
        s_sim = torch.log(torch.diag(ID_HE))

        weight_HH = l_sim.detach() / self.scale
        weight_HE = s_sim.detach() / self.scale
        wt = 2 * weight_HE * weight_HH / (weight_HE + weight_HH + 1e-8)
        wt[weight_HH < 0] = 0

        both_sim = l_sim * wt + s_sim * (1 - wt)
        adaptive_pos = torch.diag(torch.exp(both_sim))

        ID_ada = ID_HE.clone()
        ID_ada[range(B), range(B)] = torch.diag(adaptive_pos).to(ID_ada.dtype)
        ID_ada_L1 = F.normalize(ID_ada, p=1, dim=1)

        # ------------------------------
        # Final Loss
        # ------------------------------
        loss_HH = -torch.log(torch.diag(ID_HH_L1)).mean()
        loss_HE = -torch.log(torch.diag(ID_HE_L1)).mean()
        loss_ada = -torch.log(torch.diag(ID_ada_L1)).mean()

        if self.loss_type == 'sp-h':
            return loss_HH
        elif self.loss_type == 'sp-lh':
            return loss_HE
        elif self.loss_type == 'adasp':
            return loss_ada
        else:
            raise ValueError("Invalid loss_type.")


class CLIPLoss(nn.Module):
    """
    Standard Contrastive Language-Image Pre-training (CLIP) Loss.
    
    This loss function encourages the model to align the representations of 
    matched image-text pairs while pushing apart unmatched pairs within the batch.
    It is symmetric, calculating both Image-to-Text and Text-to-Image losses.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature (float): A scaling factor for the logits. 
                                 Lower values make the distribution sharper (more confident).
                                 Default is often 0.07 or learnable in the original paper.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeddings (Tensor): Shape [Batch_Size, Feature_Dim].
            text_embeddings (Tensor): Shape [Batch_Size, Feature_Dim].

        Returns:
            Tensor: Scalar loss value.
        """
        # 1. L2 Normalization
        # Normalize features to the unit hypersphere so that the dot product 
        # is equivalent to Cosine Similarity.
        # dim=-1 ensures we normalize along the feature dimension.
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # 2. Compute Similarity Matrix (Logits)
        # Perform matrix multiplication: [B, D] @ [D, B] -> [B, B]
        # Element (i, j) represents the similarity between Image_i and Text_j.
        logits = (image_embeddings @ text_embeddings.T) / self.temperature

        # 3. Create Ground Truth Labels
        # Since Image_i matches Text_i, the correct labels are the diagonal elements.
        # labels = [0, 1, 2, ..., Batch_Size-1]
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=image_embeddings.device).long()

        # 4. Compute Symmetric Cross-Entropy Loss
        
        # Loss A: Image-to-Text (Row-wise)
        # For each image, which text is the correct match?
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Loss B: Text-to-Image (Column-wise)
        # For each text, which image is the correct match?
        # We transpose the logits to treat columns as classes.
        loss_t2i = F.cross_entropy(logits.T, labels)

        # 5. Average the losses
        return (loss_i2t + loss_t2i) / 2

    def __str__(self):
        return f"{type(self).__name__}(temperature={self.temperature})"


class LocalTokenAlignmentLoss(nn.Module):
    """
    Fine-grained Image-Text Alignment Loss (Token-to-Patch Alignment).
    
    This loss encourages the model to learn the correspondence between specific 
    text tokens (words) and specific image regions (patches).
    
    Mechanism:
    1. Attention: Use text tokens as queries to attend to image patches.
    2. Reconstruction: Aggregate image patches to form "visual-word" representations.
    3. Contrastive Learning: Ensure the "visual-word" matches its corresponding 
       original text token, while differentiating it from other tokens in the sequence.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature (float): Scaling factor for the logits. 
                                 Controls the sharpness of the distribution.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, local_image_embed: torch.Tensor, local_text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_image_embed (Tensor): Shape [Batch_Size, Num_Patches, Dim] 
                                        e.g., (B, 196, 768)
            local_text_embed (Tensor):  Shape [Batch_Size, Num_Tokens, Dim] 
                                        e.g., (B, 128, 768)

        Returns:
            Tensor: Scalar loss value.
        """
        batch_size, num_patches, dim = local_image_embed.shape
        _, num_tokens, _ = local_text_embed.shape
        device = local_image_embed.device

        # =================================================================
        # 1. Cross-Modal Attention (Grounding)
        # =================================================================
        
        # Calculate similarity between every word and every image patch.
        # Query: Text Tokens, Key: Image Patches.
        # Shape: [B, Num_Tokens, Num_Patches]
        attn_sim = torch.bmm(local_text_embed, local_image_embed.transpose(1, 2))
        
        # Scale by sqrt(dim) for stability (Scaled Dot-Product Attention)
        attn_sim = attn_sim / math.sqrt(dim)
        
        # Softmax over image patches (dim=-1) to get attention weights.
        # Interpretation: For word i, which patches are important?
        attn_scores = F.softmax(attn_sim, dim=-1)

        # =================================================================
        # 2. Visual Feature Aggregation (Reconstruction)
        # =================================================================
        
        # Weighted sum of image patches based on attention scores.
        # This creates a "Visual-Word" representation for each text token.
        # Shape: [B, Num_Tokens, Dim]
        visual_word_embed = torch.bmm(attn_scores, local_image_embed)

        # =================================================================
        # 3. Contrastive Alignment (Word-Level)
        # =================================================================

        # L2 Normalize both embeddings for Cosine Similarity
        visual_word_embed = F.normalize(visual_word_embed, p=2, dim=-1)
        local_text_embed = F.normalize(local_text_embed, p=2, dim=-1)

        # Calculate similarity between Original Text Tokens and Reconstructed Visual Tokens.
        # Note: This is done strictly WITHIN each sample (intra-sample contrastive).
        # We want the i-th text token to match the i-th visual token.
        # Shape: [B, Num_Tokens, Num_Tokens]
        # Element (b, i, j): Similarity between text-token i and visual-token j in batch b.
        word_sim_matrix = torch.bmm(local_text_embed, visual_word_embed.transpose(1, 2))
        
        # Apply temperature scaling
        word_sim_matrix = word_sim_matrix / self.temperature

        # =================================================================
        # 4. Compute Loss
        # =================================================================
        
        # The target for text token i is visual token i.
        # Targets: [0, 1, 2, ..., Num_Tokens-1] repeated for each sample in batch.
        # Shape: [Batch_Size * Num_Tokens]
        targets = torch.arange(num_tokens, device=device).long().repeat(batch_size)

        # Flatten the batch dimension to treat all tokens across the batch as independent samples
        # for the cross_entropy function (or just handle batch-wise).
        # Reshape: [B, N, N] -> [B*N, N]
        # This means for each word, we classify which visual-word in the sentence is its pair.
        sim_flat_row = word_sim_matrix.reshape(batch_size * num_tokens, num_tokens)
        
        # Symmetric: Transpose to classify which text-word corresponds to each visual-word.
        # Reshape: [B, N, N] -> [B*N, N] (after transposing the N x N blocks)
        sim_flat_col = word_sim_matrix.transpose(1, 2).reshape(batch_size * num_tokens, num_tokens)

        # Loss 1: Text-to-Visual Alignment
        loss_t2v = F.cross_entropy(sim_flat_row, targets)
        
        # Loss 2: Visual-to-Text Alignment
        loss_v2t = F.cross_entropy(sim_flat_col, targets)

        # Average the symmetric losses
        return (loss_t2v + loss_v2t) / 2.0

    def __str__(self):
        return f"{type(self).__name__}(temperature={self.temperature})"


class SoftCLIPLoss(nn.Module):
    """
    Soft CLIP Loss with Label Sharpening and Learnable Temperature.
    
    This loss function is designed for scenarios where "Many-to-Many" or "Soft" relationships 
    exist between images and texts (e.g., shared attributes or classes).
    
    Key Improvements for Convergence:
    1. Learnable Temperature: Allows the model to dynamically adjust prediction confidence.
    2. Label Sharpening: Prevents the target distribution from being too flat, 
       which often causes gradients to vanish in soft-label training.
    """

    def __init__(self, init_temp: float = 0.07, label_temp: float = 0.05):
        """
        Args:
            init_temp (float): Initial value for the learnable temperature (tau). 
                               Default 0.07 corresponds to CLIP's default.
            label_temp (float): Temperature for sharpening the Ground Truth Label distribution.
                                Lower values (e.g., 0.05) make the target distribution sharper 
                                (closer to one-hot), providing stronger supervision signals.
        """
        super().__init__()
        
        # 1. Learnable Logit Scale (Inverse Temperature)
        # We store log(1/temp) for numerical stability during training.
        # Initialized to log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        
        # 2. Fixed Label Temperature for Sharpening
        self.label_temp = label_temp

    def forward(self, 
                image_embeddings: torch.Tensor, 
                text_embeddings: torch.Tensor, 
                image_labels: torch.Tensor, 
                text_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeddings (Tensor): (Batch_Size, Embed_Dim)
            text_embeddings (Tensor): (Batch_Size, Embed_Dim)
            image_labels (Tensor): (Batch_Size, Label_Dim). Multi-hot or Class Vectors.
            text_labels (Tensor): (Batch_Size, Label_Dim). Usually same as image_labels.

        Returns:
            Tensor: Scalar loss value.
        """
        
        # =================================================================
        # 1. Feature Processing
        # =================================================================
        # Normalize features to hypersphere so dot product = cosine similarity
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Clamp the learnable logit scale to prevent gradient explosion (max temp=100)
        logit_scale = self.logit_scale.exp().clamp(max=50)
        
        # Compute Prediction Logits: Similarity between Images and Texts
        # Shape: (Batch_Size, Batch_Size)
        logits_img_txt = (image_embeddings @ text_embeddings.T) * logit_scale

        # =================================================================
        # 2. Target Generation (Label Sharpening)
        # =================================================================
        # Normalize labels to ensure the similarity calculation is consistent (0 to 1 range)
        image_labels = F.normalize(image_labels.float(), p=2, dim=-1)
        text_labels = F.normalize(text_labels.float(), p=2, dim=-1)
        
        # Compute Label Similarity Matrix (The "Soft" Ground Truth)
        # Shape: (Batch_Size, Batch_Size)
        sim_labels = image_labels @ text_labels.T
        
        # Sharpening:
        # If we just apply Softmax directly to sim_labels (values 0~1), the distribution 
        # will be very flat (high entropy), making the loss hard to minimize.
        # We divide by a small `label_temp` to peak the distribution around the true matches.
        target_probs = F.softmax(sim_labels / self.label_temp, dim=1)

        # =================================================================
        # 3. Compute Symmetric Loss
        # =================================================================
        # Loss A: Image-to-Text (Row-wise)
        loss_i2t = self._soft_xent_loss(logits_img_txt, target_probs)
        
        # Loss B: Text-to-Image (Column-wise)
        loss_t2i = self._soft_xent_loss(logits_img_txt.T, target_probs)

        return (loss_i2t + loss_t2i) / 2

    def _soft_xent_loss(self, input_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes Cross Entropy with soft targets (KL Divergence equivalent).
        Formula: - sum(P_target * log(P_pred))
        """
        # Calculate log probabilities from input logits
        # log_softmax is more numerically stable than log(softmax(x))
        logprobs = F.log_softmax(input_logits, dim=1)
        
        # Element-wise multiplication and summation over class dimension (dim=1)
        loss = -(target_probs * logprobs).sum(dim=1)
        
        # Average over the batch
        return loss.mean()

    def __str__(self):
        return f"{type(self).__name__}(init_temp={1/np.exp(self.logit_scale.item()):.3f}, label_temp={self.label_temp})"
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectTokenPatchContrastiveLoss(nn.Module):
    """
    No-attention local contrastive loss.
    - Text anchors vs all patches in batch (t2i)
    - Patch anchors vs all tokens in batch (i2t)
    Positives are all items from the matched paired sample (multi-positive InfoNCE).
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        proj_dim: int | None = None,
        temperature: float = 0.07,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

        if proj_dim is None:
            if text_dim != image_dim:
                raise ValueError(
                    f"proj_dim=None requires text_dim==image_dim, got {text_dim} vs {image_dim}"
                )
            self.text_proj = nn.Identity()
            self.image_proj = nn.Identity()
            self.out_dim = text_dim
        else:
            self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
            self.image_proj = nn.Linear(image_dim, proj_dim, bias=False)
            self.out_dim = proj_dim

    @staticmethod
    def _masked_logsumexp(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        # x: [..., K], mask: same shape, bool
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~mask, neg_inf)
        return torch.logsumexp(x_masked, dim=dim)

    def forward(
        self,
        local_image_embed: torch.Tensor,  # [B, Np, Di]
        local_text_embed: torch.Tensor,   # [B, Nt, Dt]
        image_mask: torch.Tensor | None = None,  # [B, Np], True=valid
        text_mask: torch.Tensor | None = None,   # [B, Nt], True=valid
    ) -> torch.Tensor:
        B, Np, _ = local_image_embed.shape
        _, Nt, _ = local_text_embed.shape
        device = local_image_embed.device

        if image_mask is None:
            image_mask = torch.ones(B, Np, dtype=torch.bool, device=device)
        if text_mask is None:
            text_mask = torch.ones(B, Nt, dtype=torch.bool, device=device)

        # 1) project + normalize
        img = F.normalize(self.image_proj(local_image_embed), dim=-1)  # [B, Np, D]
        txt = F.normalize(self.text_proj(local_text_embed), dim=-1)    # [B, Nt, D]

        # flatten candidates
        img_all = img.reshape(B * Np, self.out_dim)         # [B*Np, D]
        txt_all = txt.reshape(B * Nt, self.out_dim)         # [B*Nt, D]
        img_all_valid = image_mask.reshape(B * Np)          # [B*Np]
        txt_all_valid = text_mask.reshape(B * Nt)           # [B*Nt]

        # sample id for each flattened element
        img_sid = torch.arange(B, device=device).repeat_interleave(Np)  # [B*Np]
        txt_sid = torch.arange(B, device=device).repeat_interleave(Nt)  # [B*Nt]

        # ---------------- t2i ----------------
        logits_t2i = torch.matmul(txt, img_all.t()) / self.temperature   # [B, Nt, B*Np]

        same_sample_t2i = (
            torch.arange(B, device=device)[:, None, None] == img_sid[None, None, :]
        )  # [B,1,B*Np] -> broadcast to [B,Nt,B*Np]
        pos_mask_t2i = same_sample_t2i & img_all_valid[None, None, :]
        all_mask_t2i = img_all_valid[None, None, :].expand(B, Nt, -1)

        log_pos_t2i = self._masked_logsumexp(logits_t2i, pos_mask_t2i, dim=-1)  # [B,Nt]
        log_all_t2i = self._masked_logsumexp(logits_t2i, all_mask_t2i, dim=-1)  # [B,Nt]
        loss_t2i_mat = -(log_pos_t2i - log_all_t2i)                               # [B,Nt]

        valid_anchor_t2i = text_mask & pos_mask_t2i.any(dim=-1)  # [B,Nt]
        loss_t2i = (loss_t2i_mat * valid_anchor_t2i.float()).sum() / (
            valid_anchor_t2i.float().sum().clamp_min(self.eps)
        )

        # ---------------- i2t ----------------
        logits_i2t = torch.matmul(img, txt_all.t()) / self.temperature   # [B, Np, B*Nt]

        same_sample_i2t = (
            torch.arange(B, device=device)[:, None, None] == txt_sid[None, None, :]
        )  # [B,1,B*Nt]
        pos_mask_i2t = same_sample_i2t & txt_all_valid[None, None, :]
        all_mask_i2t = txt_all_valid[None, None, :].expand(B, Np, -1)

        log_pos_i2t = self._masked_logsumexp(logits_i2t, pos_mask_i2t, dim=-1)  # [B,Np]
        log_all_i2t = self._masked_logsumexp(logits_i2t, all_mask_i2t, dim=-1)  # [B,Np]
        loss_i2t_mat = -(log_pos_i2t - log_all_i2t)                               # [B,Np]

        valid_anchor_i2t = image_mask & pos_mask_i2t.any(dim=-1)  # [B,Np]
        loss_i2t = (loss_i2t_mat * valid_anchor_i2t.float()).sum() / (
            valid_anchor_i2t.float().sum().clamp_min(self.eps)
        )

        return 0.5 * (loss_t2i + loss_i2t)

    def __str__(self):
        return f"{type(self).__name__}(temperature={self.temperature}, out_dim={self.out_dim})"


class SupervisedCLIPLoss(nn.Module):
    """
    Supervised CLIP Loss with Label Sharpening and Learnable Temperature.
    
    This loss function is designed for scenarios where "Many-to-Many" or "Soft" relationships 
    exist between images and texts (e.g., shared attributes or classes).
    
    Key Improvements for Convergence:
    1. Learnable Temperature: Allows the model to dynamically adjust prediction confidence.
    2. Label Sharpening: Prevents the target distribution from being too flat, 
       which often causes gradients to vanish in soft-label training.
    """

    def __init__(self, num_classes: int, init_temp: float = 0.07, label_temp: float = 0.05):
        """
        Args:
            init_temp (float): Initial value for the learnable temperature (tau). 
                               Default 0.07 corresponds to CLIP's default.
            label_temp (float): Temperature for sharpening the Ground Truth Label distribution.
                                Lower values (e.g., 0.05) make the target distribution sharper 
                                (closer to one-hot), providing stronger supervision signals.
        """
        super().__init__()
        
        self.num_classes = num_classes
        # 1. Learnable Logit Scale (Inverse Temperature)
        # We store log(1/temp) for numerical stability during training.
        # Initialized to log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        
        # 2. Fixed Label Temperature for Sharpening
        self.label_temp = label_temp

    def forward(self, 
                image_embeddings: torch.Tensor, 
                text_embeddings: torch.Tensor, 
                image_labels: torch.Tensor, 
                text_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeddings (Tensor): (Batch_Size, Embed_Dim)
            text_embeddings (Tensor): (Batch_Size, Embed_Dim)
            image_labels (Tensor): (Batch_Size, Label_Dim). Multi-hot or Class Vectors.
            text_labels (Tensor): (Batch_Size, Label_Dim). Usually same as image_labels.

        Returns:
            Tensor: Scalar loss value.
        """
        
        # =================================================================
        # 1. Feature Processing
        # =================================================================
        # Normalize features to hypersphere so dot product = cosine similarity
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Clamp the learnable logit scale to prevent gradient explosion (max temp=100)
        logit_scale = self.logit_scale.exp().clamp(max=50)
        
        # Compute Prediction Logits: Similarity between Images and Texts
        # Shape: (Batch_Size, Batch_Size)
        logits_img_txt = (image_embeddings @ text_embeddings.T) * logit_scale

        # =================================================================
        # 2. Target Generation (Label Sharpening)
        # =================================================================
        # Normalize labels to ensure the similarity calculation is consistent (0 to 1 range)
        image_labels = image_labels[:, :self.num_classes]
        text_labels = text_labels[:, :self.num_classes]
        image_labels = F.normalize(image_labels.float(), p=2, dim=-1)
        text_labels = F.normalize(text_labels.float(), p=2, dim=-1)
        
        # Compute Label Similarity Matrix (The "Soft" Ground Truth)
        # Shape: (Batch_Size, Batch_Size)
        sim_labels = image_labels @ text_labels.T
        
        # Sharpening:
        # If we just apply Softmax directly to sim_labels (values 0~1), the distribution 
        # will be very flat (high entropy), making the loss hard to minimize.
        # We divide by a small `label_temp` to peak the distribution around the true matches.
        target_probs = F.softmax(sim_labels / self.label_temp, dim=1)

        # =================================================================
        # 3. Compute Symmetric Loss
        # =================================================================
        # Loss A: Image-to-Text (Row-wise)
        loss_i2t = self._soft_xent_loss(logits_img_txt, target_probs)
        
        # Loss B: Text-to-Image (Column-wise)
        # Note: We transpose logits and targets to align predictions with ground truth
        loss_t2i = self._soft_xent_loss(logits_img_txt.T, target_probs.T)

        return (loss_i2t + loss_t2i) / 2

    def _soft_xent_loss(self, input_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes Cross Entropy with soft targets (KL Divergence equivalent).
        Formula: - sum(P_target * log(P_pred))
        """
        # Calculate log probabilities from input logits
        # log_softmax is more numerically stable than log(softmax(x))
        logprobs = F.log_softmax(input_logits, dim=1)
        
        # Element-wise multiplication and summation over class dimension (dim=1)
        loss = -(target_probs * logprobs).sum(dim=1)
        
        # Average over the batch
        return loss.mean()

    def __str__(self):
        return f"{type(self).__name__}(init_temp={1/np.exp(self.logit_scale.item()):.3f}, label_temp={self.label_temp})"