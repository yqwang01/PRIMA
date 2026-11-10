import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import gc

class MLPProjector(nn.Module):
    """
    Simple MLP for global features (CLS token or Text).
    Structure: Norm -> Linear -> GELU -> Dropout -> Linear
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(output_dim, output_dim)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

class ConvProjector(nn.Module):
    """
    [NEW] Convolutional Projector for Patch Tokens.
    Downsamples visual tokens to reduce sequence length (e.g., 256 -> 64).
    """
    def __init__(self, input_dim, output_dim, downsample_rate=2, padding=0):
        super().__init__()
        # Conv2d with stride=2 reduces H and W by half (Total tokens / 4)
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=downsample_rate, stride=downsample_rate, padding=padding),
            nn.GELU(),
            # nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=2, padding=0),
            # nn.GELU(),
            nn.Flatten(2) # Flatten H,W dimensions -> [B*M, C, N]
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        # Input x: [Batch, N_patches, Dim]
        # We assume square grid for patches (e.g., 16x16=256)
        B, M, N, D = x.shape
        H = W = int(N ** 0.5) 
        
        # Reshape to image format [B, Dim, H, W] for Conv2d
        x = x.permute(0, 1, 3, 2).view(B*M, D, H, W)
        
        # Apply Conv Downsampling
        x = self.net(x) # -> [B*M, Out_Dim, N_new]
        
        # Reshape back to sequence [B, N_new, Out_Dim]
        x = x.transpose(1, 2)
        x = x.view(B, M, -1, self.output_dim)
        x = self.norm(x)
        return x

class TextConvProjector(nn.Module):
    """
    1D Convolutional Projector for Text Sequences (e.g., BERT output).
    Compresses a long sequence (e.g., 512 tokens) into a shorter one to save VRAM.
    """
    def __init__(self, input_dim, output_dim, downsample_rate=2):
        super().__init__()
        
        # 1D Convolution: Reducing sequence length via stride
        # Kernel size = stride ensures non-overlapping windows (like pooling)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=downsample_rate, stride=downsample_rate),
            nn.GELU()
        )
        # Normalization applied after projecting to the LLM dimension
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Input x: [Batch, Seq_Len, Dim] (e.g., [B, 512, 768])
        
        # 1. Transpose for Conv1d: [B, Dim, Seq_Len]
        x = x.transpose(1, 2)
        
        # 2. Conv compression: [B, Out_Dim, Seq_Len // rate]
        x = self.net(x)
        
        # 3. Transpose back: [B, Seq_Len_New, Out_Dim]
        x = x.transpose(1, 2)
        
        return self.norm(x)

class FeatureFusionLLM(nn.Module):
    def __init__(self, base_model_path, class_names, img_feat_dim=768, text_feat_dim=768, feat_dropout=0.2, noise_scale=0.05,
    downsample_rate=2, padding=0, downsample_rate_text=2, use_patch_tokens=True,
    separate_text_cls=False):
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.noise_scale = noise_scale
        self.use_patch_tokens = use_patch_tokens
        self.separate_text_cls = separate_text_cls
        print(f"Initializing FeatureFusionLLM with {self.num_classes} classes.")

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ==========================================================
        # Step A: Add Special Tokens
        # ==========================================================
        special_tokens_dict = {'additional_special_tokens': [
            "<|img_start|>", "<|img_end|>", 
            "<|txt_start|>", "<|txt_end|>"
        ]}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # 2. Load Base LLM
        print(f"Loading base model from {base_model_path}...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=dtype, trust_remote_code=True
        )

        # Resize Embedding & Smart Init
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm_dim = self.llm.config.hidden_size if hasattr(self.llm.config, "hidden_size") else self.llm.model.embed_tokens.weight.shape[1]
        
        with torch.no_grad():
            embedding_layer = self.llm.get_input_embeddings()
            init_map = {
                "<|img_start|>": "image", "<|img_end|>": "end",
                "<|txt_start|>": "text", "<|txt_end|>": "end"
            }
            print("[Smart Init] Initializing special tokens...")
            for new_tok, old_word in init_map.items():
                new_id = self.tokenizer.convert_tokens_to_ids(new_tok)
                old_ids = self.tokenizer.encode(old_word, add_special_tokens=False)
                if len(old_ids) > 0:
                    embedding_layer.weight[new_id] = embedding_layer.weight[old_ids[0]]

        # Cache IDs
        self.id_img_s = self.tokenizer.convert_tokens_to_ids("<|img_start|>")
        self.id_img_e = self.tokenizer.convert_tokens_to_ids("<|img_end|>")
        self.id_txt_s = self.tokenizer.convert_tokens_to_ids("<|txt_start|>")
        self.id_txt_e = self.tokenizer.convert_tokens_to_ids("<|txt_end|>")

        # 3. Projectors (Updated for CLS + Patch architecture)
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        
        # Projector for the CLS token (Global info)
        self.img_cls_projector = MLPProjector(img_feat_dim, self.llm_dim).to(dtype=dtype)
        
        # [NEW] Conv Projector for Patch tokens (Local info, downsampled)
        # downsample_rate=2 means 256 tokens -> 64 tokens
        self.img_patch_projector = ConvProjector(img_feat_dim, self.llm_dim, downsample_rate=downsample_rate, padding=padding).to(dtype=dtype)
        
        self.text_projector = TextConvProjector(text_feat_dim, self.llm_dim, downsample_rate=downsample_rate_text).to(dtype=dtype)
        if self.separate_text_cls:
            self.text_cls_projector = MLPProjector(
                text_feat_dim, self.llm_dim
            ).to(dtype=dtype)
        
        # 4. Prompt Setup
        options_str = ", ".join(class_names)
        self.prompt_text = f"Based on the preceding features (bounded by special tokens), classify the diagnosis. Options: {options_str}. Answer: "
        prompt_ids = self.tokenizer(self.prompt_text, return_tensors="pt").input_ids
        if prompt_ids.shape[1] > 1 and prompt_ids[0, 0] == self.tokenizer.bos_token_id:
            prompt_ids = prompt_ids[:, 1:]
        
        with torch.no_grad():
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        self.register_buffer("fixed_prompt_embeds", prompt_embeds)

        # 5. Target Token IDs
        self.target_token_ids = []
        for name in class_names:
            word_with_space = " " + name.strip()
            ids = self.tokenizer.encode(word_with_space, add_special_tokens=False)
            self.target_token_ids.append(ids[0])
        self.target_token_ids = torch.tensor(self.target_token_ids, device=self.llm.device)

    def forward(self, img_feats, text_feats):
        # img_feats expected shape: [Batch, N_tokens, Dim] (DINO output: 1 CLS + 256 Patches)
        batch_size = img_feats.shape[0]
        device = img_feats.device
        target_dtype = self.llm.dtype
        
        # 1. Feature Augmentation (Noise Injection)
        if self.training:
            img_feats = img_feats + torch.randn_like(img_feats) * self.noise_scale
            text_feats = text_feats + torch.randn_like(text_feats) * self.noise_scale
            
        img_feats = self.feat_dropout(img_feats)
        text_feats = self.feat_dropout(text_feats)
        
        # 2. Split and Project Images
        # Assuming index 0 is CLS, 1: end are Patches
        img_cls = img_feats[:, :, 0, :]   # [Batch, Dim]
        img_patches = img_feats[:, :, 1:, :] # [Batch, N_patches, Dim]
        
        # Project CLS (MLP) -> [Batch, 1, LLM_Dim]
        img_cls_embeds = self.img_cls_projector(img_cls.to(target_dtype))
        if self.use_patch_tokens:
            # Project Patches (Conv) -> [Batch, M, N_new, LLM_Dim]
            img_patch_embeds = self.img_patch_projector(img_patches.to(target_dtype))
            img_patch_embeds = img_patch_embeds.view(batch_size, -1, self.llm_dim) # [Batch, M * N_new, LLM_Dim]
            # Concatenate: [CLS] + [Downsampled Patches]
            img_embeds = torch.cat([img_cls_embeds, img_patch_embeds], dim=1)
        else:
            img_embeds = img_cls_embeds
        
        # Project text while preserving the global CLS token separately.
        text_feats = text_feats.to(target_dtype)
        if self.separate_text_cls:
            text_cls_feats = text_feats[:, 0:1, :]
            text_local_feats = text_feats[:, 1:, :]
            text_cls_embeds = self.text_cls_projector(text_cls_feats)
            text_local_embeds = self.text_projector(text_local_feats)
            text_embeds = torch.cat(
                [text_cls_embeds, text_local_embeds], dim=1
            )
        else:
            text_embeds = self.text_projector(text_feats)
        # 3. Prepare Special Tokens
        embed_layer = self.llm.get_input_embeddings()
        img_s = embed_layer(torch.tensor([self.id_img_s], device=device)).expand(batch_size, 1, -1)
        img_e = embed_layer(torch.tensor([self.id_img_e], device=device)).expand(batch_size, 1, -1)
        txt_s = embed_layer(torch.tensor([self.id_txt_s], device=device)).expand(batch_size, 1, -1)
        txt_e = embed_layer(torch.tensor([self.id_txt_e], device=device)).expand(batch_size, 1, -1)
        prompt = self.fixed_prompt_embeds.expand(batch_size, -1, -1)
        
        # 4. Concatenate All
        inputs_embeds = torch.cat([
            img_s, img_embeds, img_e,
            txt_s, text_embeds, txt_e,
            prompt
        ], dim=1)
        
        # 5. LLM Forward (Logits only)
        outputs = self.llm(inputs_embeds=inputs_embeds, return_dict=True)
        last_token_logits = outputs.logits[:, -1, :]

        # 6. Extract Class Scores
        class_logits_list = [last_token_logits[:, tid] for tid in self.target_token_ids]
        class_logits = torch.stack(class_logits_list, dim=1) 

        return last_token_logits, class_logits


def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], use_lora=True):
    # 1. Apply LoRA
    if use_lora:
        print(f"Strategy: LoRA (r={r}, alpha={lora_alpha}, dropout={lora_dropout})")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        model.llm = get_peft_model(model.llm, peft_config)
    else:
        print("Strategy: Linear Probing (LLM Frozen)")

    # 2. Freeze ALL first
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze LoRA
    if use_lora:
        for name, param in model.llm.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    # [Smart Init] Freeze Embeddings
    model.llm.get_input_embeddings().weight.requires_grad = False 
    print("Embedding layer is FROZEN (Smart Init used).")

    # 4. Unfreeze All Projectors
    for param in model.img_cls_projector.parameters():
        param.requires_grad = True
    for param in model.img_patch_projector.parameters():
        param.requires_grad = True
    for param in model.text_projector.parameters():
        param.requires_grad = True
    if getattr(model, "separate_text_cls", False):
        for param in model.text_cls_projector.parameters():
            param.requires_grad = True
    
    # Stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} ({100 * trainable / total:.2f}%)")
    
    return model
