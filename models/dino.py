import torch
import torch.nn as nn
from typing import Union, List, Dict, Optional
from transformers import AutoConfig, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class DINOv3ImageEncoder(nn.Module):
    """
    Wrapper class for DINOv3/DINOv2 Image Encoder with built-in LoRA (PEFT) support.
    
    This class handles:
    1. Loading pre-trained DINO models from Hugging Face.
    2. Extracting specific features (CLS token, Patch tokens).
    3. Applying Parameter-Efficient Fine-Tuning (LoRA) automatically.
    """

    def __init__(
        self, 
        model_name: str = "facebook/dinov2-base", 
        freeze: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = ["query", "value"] 
    ):
        """
        Args:
            model_name (str): Hugging Face model ID (e.g., 'facebook/dinov2-base').
            freeze (bool): If True, freezes the entire backbone (Linear Probing mode).
                           Ignored if use_lora is True.
            use_lora (bool): Whether to enable LoRA fine-tuning.
            lora_r (int): LoRA rank dimension. Higher means more parameters but potentially better adaptation.
            lora_alpha (int): LoRA scaling factor.
            lora_dropout (float): Dropout probability for LoRA layers.
            lora_target_modules (List[str]): List of module names to apply LoRA to. 
                                             For standard ViT, usually ["query", "value"] or ["q_proj", "v_proj"].
        """
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        
        print(f"Loading HF Model: {model_name}...")
        
        try:
            # 1. Load Configuration first to inspect model attributes (e.g., hidden size)
            self.config = AutoConfig.from_pretrained(model_name)
            
            # 2. Load the Base Model
            # We use AutoModel to load the encoder without a specific task head.
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
            self.embed_dim = self.config.hidden_size
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model {model_name}. Error: {e}")

        # 3. Apply LoRA (PEFT) if requested
        if self.use_lora:
            print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}, targets={lora_target_modules})...")
            
            # Define LoRA configuration
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                # IMPORTANT: Set task_type to FEATURE_EXTRACTION.
                # This prevents PEFT from looking for a classification head (logits) 
                # and ensures it works correctly with the backbone model.
                task_type=TaskType.FEATURE_EXTRACTION 
            )
            
            # Wrap the base model with PEFT.
            # This automatically freezes the original weights and makes only LoRA adapters trainable.
            self.model = get_peft_model(self.model, peft_config)
            
            # Print trainable parameters to verify LoRA application
            self.model.print_trainable_parameters()
            
        # 4. If NOT using LoRA, handle standard freezing (Linear Probing)
        elif freeze:
            self._freeze_parameters()

    def _freeze_parameters(self):
        """Helper to freeze all parameters in the model (for Linear Probing)."""
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"Model {self.model_name} parameters fully frozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Basic Mode] Returns the global CLS token embedding.
        
        Args:
            x (Tensor): Input images of shape (Batch_Size, Channels, Height, Width).
                        Pixels should be normalized using the ImageProcessor beforehand.
        
        Returns:
            Tensor: Global feature vector of shape (Batch_Size, Embed_Dim).
        """
        # Forward pass
        outputs = self.model(pixel_values=x)
        
        # Handle potential output structure differences in PEFT vs Native models
        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        else:
            # Fallback for tuple outputs
            sequence_output = outputs[0]

        # Return the CLS token (Index 0)
        return sequence_output[:, 0]

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        [Advanced Mode] Extracts separated feature maps (CLS token and Patch tokens).
        
        Args:
            x (Tensor): Input images of shape (B, C, H, W).
            
        Returns:
            Dict[str, Tensor]: Dictionary containing:
                - "x_norm_clstoken": (B, D)
                - "x_norm_patchtokens": (B, N_patches, D)
        """
        outputs = self.model(pixel_values=x)
        
        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]
        
        # Handle Register Tokens (specific to DINOv2/v3 architecture)
        # Registers are inserted between CLS and Patches: [CLS, Reg1...RegN, Patch1...PatchM]
        num_registers = getattr(self.config, "num_register_tokens", 0)
        
        # 1. Extract CLS Token (Index 0)
        cls_token = sequence_output[:, 0]
        
        # 2. Extract Patch Tokens
        # Skip CLS (index 0) and all Register tokens
        patch_tokens = sequence_output[:, 1 + num_registers :]

        return {
            "x_norm_clstoken": cls_token,       
            "x_norm_patchtokens": patch_tokens  
        }

    def save_lora_weights(self, save_path: str):
        """
        Helper method to save ONLY the LoRA adapters, not the full model.
        This saves storage space (MBs vs GBs).
        """
        if self.use_lora:
            self.model.save_pretrained(save_path)
            print(f"LoRA adapters saved to {save_path}")
        else:
            print("LoRA is not enabled, nothing to save.")