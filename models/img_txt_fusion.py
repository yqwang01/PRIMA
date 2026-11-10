from sklearn.metrics import log_loss
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel, PeftConfig, TaskType, get_peft_model, LoraConfig
from utils import load_model
import numpy as np
from models.dino import DINOv3ImageEncoder
from models.eva_vit import create_eva_vit_g
from models.vit_base import ViTEncoder
'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''


class ImgTxtFusion(nn.Module):

    def __init__(self, config, vision_encoder, text_encoder, visual_path):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.d_model = config.d_model
        self.d_visual = config.d_visual

        if "bert" in str(text_encoder).lower():
            if config.peft_model_path is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
                # self.tokenizer.model_max_length = 512
                peft_config = PeftConfig.from_pretrained(config.peft_model_path)
                if peft_config.task_type == TaskType.CAUSAL_LM:
                    print("Detected legacy CAUSAL_LM LoRA adapter for BERT; overriding task_type to FEATURE_EXTRACTION.")
                    peft_config.task_type = TaskType.FEATURE_EXTRACTION

                base_model_path = peft_config.base_model_name_or_path or text_encoder
                # This LoRA adapter is trained on an AutoModelForMaskedLM backbone
                # (module path includes ".bert.encoder..."), so we must load the same
                # backbone class to avoid adapter key mismatches.
                base_model = AutoModelForMaskedLM.from_pretrained(base_model_path)
                self.text_model = PeftModel.from_pretrained(base_model, config.peft_model_path, config=peft_config)
                self.text_model.config.output_hidden_states = True
                for name, param in self.text_model.named_parameters():
                    if "lora" in name and not config.freeze_text_layers: # type: ignore
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print("Trainable parameters in PeftModel:", [name for name, param in self.text_model.named_parameters() if param.requires_grad])
            else:
                self.text_model = self._get_bert_basemodel(text_encoder, trained_layers=config.trained_text_layers)
                self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
                # self.tokenizer.model_max_length = 512
        else:
            raise ValueError("Unsupported text encoder.")
        
        if vision_encoder == "DINOv3":
            self.visual_model = DINOv3ImageEncoder(
                model_name=config.dino_model_name, 
                freeze=config.freeze_dino_vision,
                use_lora=config.dino_use_lora,
                lora_r=config.dino_lora_r,
                lora_alpha=config.dino_lora_alpha,
                lora_dropout=config.dino_lora_dropout,
                lora_target_modules=config.dino_lora_target_modules
            )
            
            self.d_visual = self.visual_model.embed_dim
            print("DINOv3 embedding dimension:", self.d_visual)
        elif vision_encoder == "EVA_vit_g":
            self.visual_model = create_eva_vit_g(
                img_size=config.eva_img_size,
                patch_size=config.eva_patch_size,
                drop_path_rate=config.eva_drop_path_rate,
                use_checkpoint=config.eva_use_checkpoint,
                precision=config.eva_precision,
            )
            self.eva_raw_dim = self.visual_model.num_features
            self.d_visual = config.d_visual
            self.eva_precision = config.eva_precision
            self.eva_feature_compressor = nn.Linear(self.eva_raw_dim, self.d_visual)
            self._configure_eva_training(config)
            print("EVA-G raw embedding dimension:", self.eva_raw_dim)
            print("EVA-G projected embedding dimension:", self.d_visual)
        elif vision_encoder == "ViT_base":
            self.visual_model = ViTEncoder(
                model_name="vit_base",
                image_size=config.vit_base_img_size,
                pretrained_source=config.vit_base_pretrained_source,
                pretrained_url=config.vit_base_pretrained_url,
                pretrained_path=config.vit_base_pretrained_path,
                use_grad_checkpointing=config.vit_base_use_checkpoint,
                ckpt_layer=config.vit_base_ckpt_layer,
                drop_path_rate=config.vit_base_drop_path_rate,
            )
            self.vit_base_raw_dim = self.visual_model.feature_dim
            self.d_visual = config.d_visual
            if self.vit_base_raw_dim != self.d_visual:
                self.vit_base_feature_compressor = nn.Linear(self.vit_base_raw_dim, self.d_visual)
            self._configure_vit_base_training(config)
            print("ViT Base raw embedding dimension:", self.vit_base_raw_dim)
            print("ViT Base projected embedding dimension:", self.d_visual)
        else:
            raise ValueError("Unsupported visual encoder.")
        
        # self.disease_embedding_layer = nn.Linear(768,256)
        # self.cl_fc = nn.Linear(256,768)
        
        
        # projection head
        self.image_projection = ProjectionHead(self.d_visual, self.d_model * 2, self.d_model)
        self.text_projection = ProjectionHead(self.d_model, self.d_model * 2, self.d_model)

        # layer_norm
        self.ln_1 = nn.LayerNorm(self.d_visual)
        self.ln_2 = nn.LayerNorm(self.d_model)

        # fusion module
        self.fusion_multiview = SymmetricConcatFusion(self.d_model, num_heads=config.num_fusion_heads, forward_dim_scale=config.forward_dim_scale,
                        num_layers=config.num_fusion_layers, dropout=config.fusion_dropout)
        self.classifier = nn.Linear(self.d_visual, config.num_classes)

        new_modules = [
            getattr(self, "eva_feature_compressor", None),
            getattr(self, "vit_base_feature_compressor", None),
            self.image_projection,
            self.text_projection,
            self.ln_1, 
            self.ln_2,
            self.fusion_multiview,
            self.classifier
        ]
        for module in new_modules:
            if module is not None:
                module.apply(self._init_weights)
        

    def _get_bert_basemodel(self, bert_model_name, trained_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("Text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary") # type: ignore

        for param in model.parameters():
            param.requires_grad = False
        if trained_layers is not None:
            for layer_idx in trained_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = True
        
        print(f"Trainable layers in {bert_model_name}:", [name for name, param in model.named_parameters() if param.requires_grad])
        return model


    def _configure_eva_training(self, config):
        eva_train_mode = config.eva_train_mode
        if eva_train_mode == "freeze_all":
            for param in self.visual_model.parameters():
                param.requires_grad = False
        elif eva_train_mode == "last_n":
            for param in self.visual_model.parameters():
                param.requires_grad = False
            last_n = config.eva_unfreeze_last_n_blocks
            if last_n > 0:
                for block in self.visual_model.blocks[-last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True
        elif eva_train_mode == "lora":
            peft_config = LoraConfig(
                r=config.eva_lora_r,
                lora_alpha=config.eva_lora_alpha,
                target_modules=config.eva_lora_target_modules,
                lora_dropout=config.eva_lora_dropout,
                bias="none",
            )
            self.visual_model = get_peft_model(self.visual_model, peft_config)
            self.visual_model.print_trainable_parameters()
        else:
            raise ValueError("Unknown EVA train mode: {}".format(eva_train_mode))
        print("EVA-G train mode:", eva_train_mode)
        print("Trainable layers in EVA-G:", [name for name, param in self.visual_model.named_parameters() if param.requires_grad])


    def _configure_vit_base_training(self, config):
        vit_base_train_mode = config.vit_base_train_mode
        if vit_base_train_mode == "freeze_all":
            for param in self.visual_model.parameters():
                param.requires_grad = False
        elif vit_base_train_mode == "lora":
            peft_config = LoraConfig(
                r=config.vit_base_lora_r,
                lora_alpha=config.vit_base_lora_alpha,
                target_modules=config.vit_base_lora_target_modules,
                lora_dropout=config.vit_base_lora_dropout,
                bias="none",
            )
            self.visual_model = get_peft_model(self.visual_model, peft_config)
            self.visual_model.print_trainable_parameters()
        else:
            raise ValueError("Unknown ViT Base train mode: {}".format(vit_base_train_mode))
        print("ViT Base train mode:", vit_base_train_mode)
        print("Trainable layers in ViT Base:", [name for name, param in self.visual_model.named_parameters() if param.requires_grad])
    

    def image_encoder(self, xis):
        #patch features
        batch_size = xis.shape[0]
        if self.vision_encoder == "DINOv3":
            x_features = self.visual_model.forward_features(xis)
            x_cls = x_features["x_norm_clstoken"]
            x_patches = x_features["x_norm_patchtokens"]
        elif self.vision_encoder == "EVA_vit_g":
            if self.eva_precision in ["fp16", "bf16"] and xis.is_cuda:
                _dtype = torch.float16 if self.eva_precision == "fp16" else torch.bfloat16
                with torch.cuda.amp.autocast(dtype=_dtype):
                    x_features = self.visual_model(xis)
            else:
                eva_dtype = next(self.visual_model.parameters()).dtype
                x_features = self.visual_model(xis.to(dtype=eva_dtype))
            x_features = x_features.float()
            x_features = self.eva_feature_compressor(x_features)
            x_cls = x_features[:, 0]
            x_patches = x_features[:, 1:]
        elif self.vision_encoder == "ViT_base":
            if hasattr(self.visual_model, "forward_features"):
                x_features = self.visual_model.forward_features(xis)
            else:
                x_cls, x_patches = self.visual_model(xis)
                x_features = torch.cat([x_cls.unsqueeze(1), x_patches], dim=1)
            x_features = x_features.float()
            if hasattr(self, "vit_base_feature_compressor"):
                x_features = self.vit_base_feature_compressor(x_features)
            x_cls = x_features[:, 0]
            x_patches = x_features[:, 1:]
        else:
            raise ValueError("Unsupported visual encoder.")
        
        cls_logits = self.classifier(x_cls)
        img_features = torch.cat([x_cls.unsqueeze(1), x_patches], dim=1)  # pyright: ignore[reportCallIssue, reportArgumentType]
        img_proj_features = self.ln_1(img_features)
        img_proj_features = self.image_projection(img_proj_features)

        return img_features, img_proj_features, cls_logits


    def forward(self, img_1, img_2=None, text=None):

        if img_2 is None:
            img_1_features, img_1_proj_features, logits_1 = self.image_encoder(img_1)
            return_dict = {
                "img_1_features": img_1_features,
                "img_1_proj_features": img_1_proj_features,
                "logits_1": logits_1
            }
            return return_dict
            
        else:
            img_1_features, img_1_proj_features, logits_1 = self.image_encoder(img_1)
            img_2_features, img_2_proj_features, logits_2 = self.image_encoder(img_2)
            B = img_1.shape[0]
            device = img_1.device
            
            if 'bert' in str(self.text_encoder).lower():
                text_tokens = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_tokens["input_ids"].to(device).to(torch.long) # type: ignore
                attention_mask = text_tokens["attention_mask"].to(device).to(torch.long) # type: ignore

                text_model_outputs = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if hasattr(text_model_outputs, 'last_hidden_state') and text_model_outputs.last_hidden_state is not None:
                    text_features = text_model_outputs.last_hidden_state
                elif hasattr(text_model_outputs, 'hidden_states') and text_model_outputs.hidden_states is not None:
                    text_features = text_model_outputs.hidden_states[-1]
                else:
                    raise ValueError("Unsupported text model output format.")
            else:
                raise ValueError("Unsupported text encoder.")

            text_features = self.ln_2(text_features)
            text_features = self.text_projection(text_features)
            fusion_features = self.fusion_multiview(img_1_proj_features[:, 1:, :], img_2_proj_features[:, 1:, :])
            fusion_cls = fusion_features[:, 0, :]
            fusion_patches = fusion_features[:, 1:, :]

            img_1_cls, img_1_patches = img_1_proj_features[:, 0, :], img_1_proj_features[:, 1:, :]
            img_2_cls, img_2_patches = img_2_proj_features[:, 0, :], img_2_proj_features[:, 1:, :]
            text_cls, text_patches = text_features[:, 0, :], text_features[:, 1:, :]

            return_dict = {
                "img_1_cls": img_1_cls,
                "img_1_patches": img_1_patches,
                "img_2_cls": img_2_cls,
                "img_2_patches": img_2_patches,
                "text_cls": text_cls,
                "text_patches": text_patches,
                "fusion_cls": fusion_cls,
                "fusion_patches": fusion_patches,
                "cls_logits_1": logits_1,
                "cls_logits_2": logits_2
            }

            return return_dict


    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            if module.out_proj.weight is not None:
                module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
        
        elif isinstance(module, nn.TransformerEncoderLayer):
            module.self_attn.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.self_attn.out_proj.weight.data.normal_(mean=0.0, std=0.02)
            if module.self_attn.in_proj_bias is not None:
                module.self_attn.in_proj_bias.data.zero_()
            if module.self_attn.out_proj.bias is not None:
                module.self_attn.out_proj.bias.data.zero_()
            
            module.linear1.weight.data.normal_(mean=0.0, std=0.02)
        


class SymmetricConcatFusion(nn.Module):
    """
    Fusion module using Concatenation + Transformer Self-Attention.
    
    Mechanism:
    1.  Add learnable segment embeddings to distinguish View 1 from View 2.
    2.  Concatenate [CLS] token, View 1 tokens, and View 2 tokens into a long sequence.
    3.  Apply Self-Attention. Since Q, K, and V all come from the combined sequence, 
        View 1 and View 2 interact symmetrically and bidirectionally.
    4.  Extract the [CLS] token as the final fused representation.
    """
    def __init__(self, embed_dim, num_heads=8, forward_dim_scale=4, num_layers=1, dropout=0.1):
        super().__init__()
        
        # Learnable [CLS] token to aggregate global information
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer Encoder Layer
        # batch_first=True ensures input/output shape is (Batch, Seq_Len, Dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * forward_dim_scale,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-norm is generally more stable for ViTs
        )
        
        # Stack multiple layers if needed (usually 1 or 2 is enough for fusion)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final LayerNorm before output (optional, but recommended)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, view1, view2):
        """
        Args:
            view1: Features from the first view. Shape: (Batch_Size, Num_Patches, Dim)
            view2: Features from the second view. Shape: (Batch_Size, Num_Patches, Dim)
            
        Returns:
            fused_embedding: Shape (Batch_Size, Dim)
        """
        B, N, D = view1.shape
        
        # 2. Prepare [CLS] token
        # Expand to match batch size: (1, 1, D) -> (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 3. Concatenate all tokens
        # The resulting sequence length will be: 1 (CLS) + N (View1) + N (View2)
        # Shape: (B, 1 + 2N, D)
        x = torch.cat((cls_tokens, view1, view2), dim=1)
        
        # 4. Global Self-Attention
        # Information flows freely between View 1 and View 2.
        # Since it is self-attention, the operation is order-agnostic regarding 
        # the interaction capability (A can attend to B, B can attend to A).
        x = self.encoder(x)

        fused_embedding = self.norm(x)
        
        return fused_embedding


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),         
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x shape: (Batch_Size, Sequence_Length, Input_Dim) 
        return self.head(x)
