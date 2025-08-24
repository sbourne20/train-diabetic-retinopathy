import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoModel
import math

class RETFoundBackbone(nn.Module):
    """Official RETFound Vision Transformer backbone from HuggingFace."""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 pretrained_path: str = None,
                 hf_token: str = None):
        super().__init__()
        
        # Load HuggingFace token from environment if not provided
        if hf_token is None:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not hf_token:
            raise ValueError(
                "HUGGINGFACE_TOKEN not found in environment. "
                "Please set HUGGINGFACE_TOKEN in your .env file to access RETFound model."
            )
        
        try:
            # Validate HuggingFace token format
            if not hf_token.startswith('hf_'):
                raise ValueError("Invalid HuggingFace token format. Token should start with 'hf_'")
            
            # Load official RETFound model from HuggingFace with caching
            print("Loading official RETFound model from HuggingFace...")
            print("Model: YukunZhou/RETFound_mae_natureCFP (MAE-based foundation model)")
            
            # RETFound uses custom loading, not standard AutoModel
            from huggingface_hub import hf_hub_download, login
            
            # Login to HuggingFace for gated model access
            login(token=hf_token, write_permission=False)
            
            # Download model files from HuggingFace
            model_path = hf_hub_download(
                repo_id="YukunZhou/RETFound_mae_natureCFP",
                filename="RETFound_mae_natureCFP.pth",
                token=hf_token,
                cache_dir=os.path.expanduser("~/.cache/huggingface/retfound")
            )
            
            # Load RETFound model using PyTorch (Vision Transformer architecture)
            import timm
            self.retfound_model = timm.create_model(
                'vit_large_patch16_224',
                pretrained=False,
                num_classes=0,  # Remove classification head
                img_size=img_size,
                patch_size=patch_size
            )
            
            # Load the pre-trained RETFound weights (disable weights_only for PyTorch 2.8+)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle RETFound checkpoint format (following official implementation)
            if 'model' in checkpoint:
                # RETFound MAE format
                checkpoint_model = checkpoint['model']
            elif 'teacher' in checkpoint:
                # DINOv2 format  
                checkpoint_model = checkpoint['teacher']
            elif 'state_dict' in checkpoint:
                checkpoint_model = checkpoint['state_dict']
            else:
                checkpoint_model = checkpoint
            
            # Clean up key names following official RETFound approach
            cleaned_state_dict = {}
            for k, v in checkpoint_model.items():
                # Remove common prefixes
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('backbone.'):
                    k = k[9:]
                
                # Handle MLP layer name changes (from official code)
                k = k.replace("mlp.w12.", "mlp.fc1.")
                k = k.replace("mlp.w3.", "mlp.fc2.")
                
                cleaned_state_dict[k] = v
            
            # Interpolate position embeddings if necessary (from official implementation)
            self._interpolate_pos_embed(cleaned_state_dict, img_size, patch_size)
            
            # Load weights into model
            missing_keys, unexpected_keys = self.retfound_model.load_state_dict(cleaned_state_dict, strict=False)
            print(f"✅ RETFound weights loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            
            # Get model configuration (from timm ViT model)
            self.embed_dim = self.retfound_model.embed_dim
            self.num_patches = (img_size // patch_size) ** 2
            
            # Log model architecture details
            print(f"✅ Successfully loaded RETFound model:")
            print(f"   - Architecture: Vision Transformer (ViT) with RETFound pre-training")
            print(f"   - Hidden size: {self.embed_dim}")
            print(f"   - Image patches: {self.num_patches}")
            print(f"   - Model type: {type(self.retfound_model).__name__}")
            print(f"   - Weights from: {model_path}")
            print(f"   - Cached at: ~/.cache/huggingface/retfound")
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Provide specific error guidance
            if "token" in error_msg or "authentication" in error_msg:
                raise RuntimeError(
                    f"❌ HuggingFace authentication failed: {e}\n\n"
                    f"🔧 SOLUTION:\n"
                    f"1. Get a valid HuggingFace token from: https://huggingface.co/settings/tokens\n"
                    f"2. Request access to the model: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP\n"
                    f"3. Add token to .env file: HUGGINGFACE_TOKEN=hf_your_token_here\n"
                    f"4. Wait for model access approval (may take 24-48 hours)"
                )
            elif "connection" in error_msg or "network" in error_msg:
                raise RuntimeError(
                    f"❌ Network connection failed: {e}\n\n"
                    f"🔧 SOLUTION:\n"
                    f"1. Check your internet connection\n"
                    f"2. Verify firewall/proxy settings\n"
                    f"3. Try again in a few minutes"
                )
            else:
                raise RuntimeError(
                    f"❌ Failed to load RETFound model: {e}\n\n"
                    f"🔧 TROUBLESHOOTING:\n"
                    f"1. Verify HUGGINGFACE_TOKEN in .env file\n"
                    f"2. Request access: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP\n"
                    f"3. Check internet connection\n"
                    f"4. Clear cache: rm -rf ~/.cache/huggingface\n"
                    f"5. Contact: ykzhoua@gmail.com for model access issues"
                )
    
    def _interpolate_pos_embed(self, checkpoint_model: dict, img_size: int, patch_size: int):
        """Interpolate position embeddings for different image sizes (from official RETFound)."""
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = (img_size // patch_size) ** 2
            num_extra_tokens = self.retfound_model.pos_embed.shape[-2] - num_patches
            
            # Height and width for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # Height and width for the new position embedding
            new_size = int(num_patches ** 0.5)
            
            if orig_size != new_size:
                print(f"Position interpolation from {orig_size}x{orig_size} to {new_size}x{new_size}")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through official RETFound backbone."""
        try:
            # RETFound model expects standard image input
            # Input shape: [batch_size, 3, H, W]
            if x.dim() != 4 or x.shape[1] != 3:
                raise ValueError(f"Expected input shape [batch_size, 3, H, W], got {x.shape}")
            
            # Forward through RETFound ViT model (using timm)
            # timm ViT returns features without classification head when num_classes=0
            features = self.retfound_model.forward_features(x)
            
            # timm ViT forward_features returns shape: [batch_size, num_patches + 1, embed_dim]
            # where the first token is the CLS token
            
            if features.dim() != 3:
                raise ValueError(f"Expected 3D features [batch, seq, embed], got shape {features.shape}")
            
            print(f"✅ RETFound forward pass successful: {features.shape}")
            return features  # Shape: [batch_size, num_patches + 1, embed_dim]
            
        except Exception as e:
            raise RuntimeError(
                f"❌ Forward pass failed in RETFound model: {e}\n"
                f"Input shape: {x.shape if x is not None else 'None'}\n"
                f"Expected: [batch_size, 3, 224, 224] for RETFound ViT model"
            )

class MultiTaskHead(nn.Module):
    """Enhanced multi-task classification head with comprehensive clinical features."""
    
    def __init__(self, 
                 embed_dim: int = 1024,
                 num_classes_rg: int = 5,  # Updated to 5 classes (0-4, including PDR)
                 num_classes_me: int = 3,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 enable_referable: bool = True,
                 enable_confidence: bool = True,
                 enable_localization: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_attention = use_attention
        self.enable_referable = enable_referable
        self.enable_confidence = enable_confidence
        self.enable_localization = enable_localization
        
        # Shared feature processing
        self.feature_norm = nn.LayerNorm(embed_dim)
        self.shared_projection = nn.Linear(embed_dim, embed_dim)
        
        # Attention mechanism for feature aggregation
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=16,
                dropout=dropout,
                batch_first=True
            )
        
        # Primary task heads
        self.rg_head = self._create_classification_head(embed_dim, num_classes_rg, dropout)
        self.me_head = self._create_classification_head(embed_dim, num_classes_me, dropout)
        
        # Enhanced classification heads
        if enable_referable:
            self.referable_dr_head = self._create_classification_head(embed_dim, 2, dropout)  # binary
            self.sight_threatening_head = self._create_classification_head(embed_dim, 2, dropout)  # binary
        
        # PDR activity classification
        self.pdr_activity_head = self._create_classification_head(embed_dim, 3, dropout)  # absent, suspected, present
        
        # ETDRS 4-2-1 rule components
        self.hemorrhages_4q_head = self._create_classification_head(embed_dim, 2, dropout)
        self.venous_beading_2q_head = self._create_classification_head(embed_dim, 2, dropout)
        self.irma_1q_head = self._create_classification_head(embed_dim, 2, dropout)
        self.meets_421_head = self._create_classification_head(embed_dim, 2, dropout)
        
        # NPDR findings (7 total)
        self.microaneurysms_count_head = self._create_classification_head(embed_dim, 4, dropout)  # 0, 1_20, >20, unknown
        self.intraretinal_hemorrhages_severity_head = self._create_classification_head(embed_dim, 4, dropout)  # none, mild, moderate, severe
        self.hard_exudates_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.cotton_wool_spots_count_head = self._create_classification_head(embed_dim, 3, dropout)  # 0, 1_3, >3
        self.venous_beading_quadrants_head = self._create_classification_head(embed_dim, 5, dropout)  # 0, 1, 2, 3, 4
        self.irma_quadrants_head = self._create_classification_head(embed_dim, 5, dropout)  # 0, 1, 2, 3, 4
        self.venous_looping_or_reduplication_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        
        # PDR findings (6 total)
        self.nvd_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.nve_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.nvi_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.pre_or_vitreous_hemorrhage_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.fibrovascular_proliferation_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.tractional_retinal_detachment_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        
        # Detailed PDR area/activity assessment
        self.nvd_area_head = self._create_classification_head(embed_dim, 4, dropout)  # none, <1/3DD, 1/3-1DD, >1DD
        self.nve_area_head = self._create_classification_head(embed_dim, 3, dropout)  # none, <1/2DD, >=1/2DD
        self.nv_activity_head = self._create_classification_head(embed_dim, 3, dropout)  # active, regressed, unknown
        
        # Treatment/legacy findings (2)
        self.prp_scars_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.focal_laser_scars_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        
        # Macular edema status (2 total)
        self.dme_status_head = self._create_classification_head(embed_dim, 3, dropout)  # none, non_center, center
        self.etdrs_csme_head = self._create_classification_head(embed_dim, 3, dropout)  # absent, present, unknown
        
        # Localization (2 total)
        self.within_1dd_fovea_head = self._create_classification_head(embed_dim, 5, dropout)  # none, MA, hemorrhage, hard_exudate, IRMA
        self.quadrants_involved_head = self._create_classification_head(embed_dim, 6, dropout)  # none, ST, IT, SN, IN, multiple
        
        # DME fundus surrogate (1)
        self.exudate_within_1dd_fovea_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        
        # OCT features (3)
        self.cst_um_head = self._create_regression_head(embed_dim, 1, dropout)  # continuous value
        self.intraretinal_cysts_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.subretinal_fluid_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        
        # Co-pathology/confounders (5)
        self.hypertensive_retinopathy_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.retinal_vein_occlusion_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.amd_drusen_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.myopic_degeneration_head = self._create_classification_head(embed_dim, 2, dropout)  # absent, present
        self.media_opacity_severity_head = self._create_classification_head(embed_dim, 4, dropout)  # none, mild, moderate, severe
        
        # Image quality assessment (2 total)
        self.gradable_head = self._create_classification_head(embed_dim, 2, dropout)  # not gradable, gradable
        self.image_quality_head = self._create_regression_head(embed_dim, 1, dropout)  # 0-4 score
        
        # Confidence estimation
        if enable_confidence:
            self.confidence_head = self._create_regression_head(embed_dim, 1, dropout)  # 0-1 confidence
            
        # Feature localization support
        if enable_localization:
            self.lesion_detector = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 4)  # bbox coordinates
            )
        
        # Feature extraction for language model
        self.language_features = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def _create_classification_head(self, input_dim: int, num_classes: int, dropout: float) -> nn.Module:
        """Create a standard classification head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, num_classes)
        )
    
    def _create_regression_head(self, input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        """Create a regression head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, output_dim)
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with comprehensive clinical outputs.
        
        Args:
            features: [batch_size, num_patches + 1, embed_dim]
        """
        batch_size = features.shape[0]
        
        # Use CLS token for classification
        cls_token = features[:, 0, :]  # [batch_size, embed_dim]
        
        # Optional: Use attention to aggregate patch features
        if self.use_attention:
            attended_features, attention_weights = self.attention(
                cls_token.unsqueeze(1),  # Query: [batch_size, 1, embed_dim]
                features,                # Key & Value: [batch_size, num_patches + 1, embed_dim]
                features
            )
            cls_token = attended_features.squeeze(1)  # [batch_size, embed_dim]
        else:
            attention_weights = None
        
        # Normalize and project features
        features_norm = self.feature_norm(cls_token)
        shared_features = self.shared_projection(features_norm)
        
        outputs = {}
        
        # Primary classifications
        outputs['rg_logits'] = self.rg_head(shared_features)
        outputs['me_logits'] = self.me_head(shared_features)
        
        # Enhanced classifications
        if self.enable_referable:
            outputs['referable_dr_logits'] = self.referable_dr_head(shared_features)
            outputs['sight_threatening_logits'] = self.sight_threatening_head(shared_features)
        
        # PDR activity
        outputs['pdr_activity_logits'] = self.pdr_activity_head(shared_features)
        
        # ETDRS 4-2-1 rule components
        outputs['hemorrhages_4q_logits'] = self.hemorrhages_4q_head(shared_features)
        outputs['venous_beading_2q_logits'] = self.venous_beading_2q_head(shared_features)
        outputs['irma_1q_logits'] = self.irma_1q_head(shared_features)
        outputs['meets_421_logits'] = self.meets_421_head(shared_features)
        
        # NPDR findings (7 total)
        outputs['microaneurysms_count_logits'] = self.microaneurysms_count_head(shared_features)
        outputs['intraretinal_hemorrhages_severity_logits'] = self.intraretinal_hemorrhages_severity_head(shared_features)
        outputs['hard_exudates_logits'] = self.hard_exudates_head(shared_features)
        outputs['cotton_wool_spots_count_logits'] = self.cotton_wool_spots_count_head(shared_features)
        outputs['venous_beading_quadrants_logits'] = self.venous_beading_quadrants_head(shared_features)
        outputs['irma_quadrants_logits'] = self.irma_quadrants_head(shared_features)
        outputs['venous_looping_or_reduplication_logits'] = self.venous_looping_or_reduplication_head(shared_features)
        
        # PDR findings (6 total)
        outputs['nvd_logits'] = self.nvd_head(shared_features)
        outputs['nve_logits'] = self.nve_head(shared_features)
        outputs['nvi_logits'] = self.nvi_head(shared_features)
        outputs['pre_or_vitreous_hemorrhage_logits'] = self.pre_or_vitreous_hemorrhage_head(shared_features)
        outputs['fibrovascular_proliferation_logits'] = self.fibrovascular_proliferation_head(shared_features)
        outputs['tractional_retinal_detachment_logits'] = self.tractional_retinal_detachment_head(shared_features)
        
        # Detailed PDR area/activity assessment
        outputs['nvd_area_logits'] = self.nvd_area_head(shared_features)
        outputs['nve_area_logits'] = self.nve_area_head(shared_features)
        outputs['nv_activity_logits'] = self.nv_activity_head(shared_features)
        
        # Treatment/legacy findings (2)
        outputs['prp_scars_logits'] = self.prp_scars_head(shared_features)
        outputs['focal_laser_scars_logits'] = self.focal_laser_scars_head(shared_features)
        
        # Macular edema status (2 total)
        outputs['dme_status_logits'] = self.dme_status_head(shared_features)
        outputs['etdrs_csme_logits'] = self.etdrs_csme_head(shared_features)
        
        # Localization (2 total)
        outputs['within_1dd_fovea_logits'] = self.within_1dd_fovea_head(shared_features)
        outputs['quadrants_involved_logits'] = self.quadrants_involved_head(shared_features)
        
        # DME fundus surrogate (1)
        outputs['exudate_within_1dd_fovea_logits'] = self.exudate_within_1dd_fovea_head(shared_features)
        
        # OCT features (3)
        outputs['cst_um'] = self.cst_um_head(shared_features) * 1000.0  # Scale to micrometers
        outputs['intraretinal_cysts_logits'] = self.intraretinal_cysts_head(shared_features)
        outputs['subretinal_fluid_logits'] = self.subretinal_fluid_head(shared_features)
        
        # Co-pathology/confounders (5)
        outputs['hypertensive_retinopathy_logits'] = self.hypertensive_retinopathy_head(shared_features)
        outputs['retinal_vein_occlusion_logits'] = self.retinal_vein_occlusion_head(shared_features)
        outputs['amd_drusen_logits'] = self.amd_drusen_head(shared_features)
        outputs['myopic_degeneration_logits'] = self.myopic_degeneration_head(shared_features)
        outputs['media_opacity_severity_logits'] = self.media_opacity_severity_head(shared_features)
        
        # Image quality assessment (2 total)
        outputs['gradable_logits'] = self.gradable_head(shared_features)
        outputs['image_quality_score'] = torch.sigmoid(self.image_quality_head(shared_features)) * 4.0
        
        # Confidence estimation
        if self.enable_confidence:
            outputs['grading_confidence'] = torch.sigmoid(self.confidence_head(shared_features))
        
        # Localization
        if self.enable_localization:
            if attention_weights is not None:
                outputs['spatial_attention'] = attention_weights
            outputs['lesion_bbox'] = self.lesion_detector(shared_features)
        
        # Features for language generation
        outputs['language_features'] = self.language_features(shared_features)
        outputs['shared_features'] = shared_features
        
        return outputs

class MedicalReasoningModule(nn.Module):
    """Medical reasoning module for generating explanatory text."""
    
    def __init__(self, 
                 visual_dim: int = 512,
                 vocab_size: int = 50257,
                 hidden_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 max_length: int = 256):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Visual feature projection to language model dimension
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # GPT-2 based language model for medical text generation
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_length + 32,  # Extra space for visual tokens
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=50256
        )
        
        self.language_model = GPT2LMHeadModel(config)
        
        # Special tokens for medical reasoning
        self.visual_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.medical_vocab_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, 
                visual_features: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate medical reasoning text based on visual features.
        
        Args:
            visual_features: [batch_size, visual_dim]
            input_ids: [batch_size, seq_len] - for training
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - for training loss
        """
        batch_size = visual_features.shape[0]
        
        # Project visual features to language model dimension
        visual_embeds = self.visual_projection(visual_features)  # [batch_size, hidden_dim]
        visual_embeds = visual_embeds.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Add visual token embedding
        visual_embeds = visual_embeds + self.visual_token_embed
        
        if input_ids is not None:
            # Training mode: concatenate visual and text embeddings
            text_embeds = self.language_model.transformer.wte(input_ids)
            
            # Concatenate visual and text embeddings
            combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            # Create extended attention mask
            if attention_mask is not None:
                visual_attention = torch.ones(batch_size, 1, device=attention_mask.device)
                extended_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
            else:
                extended_attention_mask = torch.ones(batch_size, combined_embeds.shape[1], 
                                                   device=combined_embeds.device)
            
            # Forward through language model
            outputs = self.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                labels=labels
            )
            
            return {
                'loss': outputs.loss if labels is not None else None,
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states
            }
        
        else:
            # Inference mode: generate text from visual features
            return self.generate_medical_text(visual_embeds)
    
    def generate_medical_text(self, 
                            visual_embeds: torch.Tensor,
                            max_new_tokens: int = 150,
                            temperature: float = 0.7,
                            do_sample: bool = True) -> Dict[str, torch.Tensor]:
        """Generate medical reasoning text from visual features."""
        
        batch_size = visual_embeds.shape[0]
        device = visual_embeds.device
        
        # Start with visual embeddings
        generated_embeds = visual_embeds
        generated_ids = []
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.language_model(inputs_embeds=generated_embeds)
            logits = outputs.logits[:, -1, :]  # Next token logits
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token)
            
            # Get embedding for next token and concatenate
            next_embed = self.language_model.transformer.wte(next_token)
            generated_embeds = torch.cat([generated_embeds, next_embed], dim=1)
            
            # Stop if EOS token is generated (simplified stopping criteria)
            if torch.all(next_token == 50256):  # EOS token
                break
        
        # Convert generated tokens to sequence
        if generated_ids:
            generated_sequence = torch.cat(generated_ids, dim=1)
        else:
            generated_sequence = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        
        return {
            'generated_ids': generated_sequence,
            'generated_embeds': generated_embeds
        }

class ClinicalRuleEngine(nn.Module):
    """Engine for applying clinical rules and generating structured outputs."""
    
    def __init__(self, medical_terms_path: str):
        super().__init__()
        
        # Load medical terms schema with fallback
        try:
            with open(medical_terms_path, 'r') as f:
                self.medical_terms = json.load(f)
        except FileNotFoundError:
            # Try alternative paths if file not found
            alternative_paths = [
                os.path.join(os.getcwd(), medical_terms_path),
                os.path.join(os.path.dirname(__file__), medical_terms_path),
                medical_terms_path.replace('data/', ''),  # Try without data/ prefix
            ]
            
            loaded = False
            for alt_path in alternative_paths:
                try:
                    with open(alt_path, 'r') as f:
                        self.medical_terms = json.load(f)
                        print(f"Loaded medical terms from: {alt_path}")
                        loaded = True
                        break
                except FileNotFoundError:
                    continue
            
            # Try downloading from GCS if still not found
            if not loaded:
                try:
                    from google.cloud import storage
                    import tempfile
                    
                    # Try to download from GCS using the provided path
                    gcs_path = medical_terms_path
                    print(f"Attempting to download medical terms from: {gcs_path}")
                    
                    # Parse GCS path
                    if gcs_path.startswith('gs://'):
                        path_parts = gcs_path.replace('gs://', '').split('/')
                        bucket_name = path_parts[0]
                        blob_path = '/'.join(path_parts[1:])
                    else:
                        # This should not happen - medical_terms_path should be a valid GCS path
                        raise ValueError(f"Invalid medical terms path: {medical_terms_path}. Expected gs:// path.")
                    
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    
                    # Download to temporary file
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
                        blob.download_to_filename(tmp_file.name)
                        
                        with open(tmp_file.name, 'r') as f:
                            self.medical_terms = json.load(f)
                        
                        print(f"Successfully loaded medical terms from GCS: {gcs_path}")
                        loaded = True
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    print(f"Failed to download medical terms from GCS: {e}")
                    pass
            
            if not loaded:
                # Use default medical terms if file still not found
                print(f"Warning: Could not load medical terms from {medical_terms_path}, using defaults")
                self.medical_terms = {
                    "severity_terms": {
                        "0": ["no diabetic retinopathy", "normal retina", "no pathological changes"],
                        "1": ["mild non-proliferative diabetic retinopathy", "early signs", "microaneurysms present"],
                        "2": ["moderate non-proliferative diabetic retinopathy", "dot-blot hemorrhages", "cotton wool spots"],
                        "3": ["severe non-proliferative diabetic retinopathy", "extensive hemorrhages", "venous abnormalities"],
                        "4": ["proliferative diabetic retinopathy", "neovascularization", "vision-threatening changes"]
                    },
                    "macular_edema_terms": {
                        "0": ["no macular edema", "normal macula", "no fluid accumulation"],
                        "1": ["mild macular edema risk", "early changes", "potential fluid retention"],
                        "2": ["high macular edema risk", "significant fluid accumulation", "vision-threatening changes"]
                    },
                    "anatomical_terms": [
                        "optic disc", "macula", "fovea", "retinal vessels", "microaneurysms",
                        "hemorrhages", "exudates", "cotton wool spots", "venous beading"
                    ]
                }
            
    def apply_clinical_rules(self, model_outputs: Dict[str, torch.Tensor]) -> List[Dict]:
        """Apply clinical decision rules to model outputs."""
        
        batch_size = model_outputs['rg_logits'].shape[0]
        results = []
        
        for i in range(batch_size):
            result = self._process_single_sample(model_outputs, i)
            results.append(result)
            
        return results
    
    def _process_single_sample(self, outputs: Dict[str, torch.Tensor], idx: int) -> Dict:
        """Process a single sample with clinical rules."""
        
        # Get predictions
        rg_pred = torch.argmax(outputs['rg_logits'][idx]).item()
        me_pred = torch.argmax(outputs['me_logits'][idx]).item()
        pdr_activity = torch.argmax(outputs['pdr_activity_logits'][idx]).item() if 'pdr_activity_logits' in outputs else 0
        
        # Get probabilities for confidence
        rg_probs = F.softmax(outputs['rg_logits'][idx], dim=0).detach().cpu().numpy()
        me_probs = F.softmax(outputs['me_logits'][idx], dim=0).detach().cpu().numpy()
        
        # Apply clinical rules
        referable_dr = self._determine_referable_dr(rg_pred, me_pred)
        sight_threatening = self._determine_sight_threatening(rg_pred, me_pred, pdr_activity)
        
        # ETDRS 4-2-1 rule (handle binary classification logits)
        hemorrhages_4q = torch.softmax(outputs['hemorrhages_4q_logits'][idx], dim=0)[1].item() > 0.5 if 'hemorrhages_4q_logits' in outputs else False
        venous_beading_2q = torch.softmax(outputs['venous_beading_2q_logits'][idx], dim=0)[1].item() > 0.5 if 'venous_beading_2q_logits' in outputs else False
        irma_1q = torch.softmax(outputs['irma_1q_logits'][idx], dim=0)[1].item() > 0.5 if 'irma_1q_logits' in outputs else False
        meets_421 = hemorrhages_4q and venous_beading_2q and irma_1q
        
        # Structured output
        result = {
            "grading": {
                "icdr_severity": f"{rg_pred}_{self._get_severity_name(rg_pred)}",
                "pdr_activity": ["absent", "suspected", "present"][pdr_activity],
                "referable_DR": referable_dr,
                "sight_threatening_DR": sight_threatening,
                "grading_confidence": outputs.get('grading_confidence', [torch.tensor(0.5)])[idx].item() if 'grading_confidence' in outputs else 0.5,
                "class_probabilities": {
                    "RG0": float(rg_probs[0]) if len(rg_probs) > 0 else 0.0,
                    "RG1": float(rg_probs[1]) if len(rg_probs) > 1 else 0.0,
                    "RG2": float(rg_probs[2]) if len(rg_probs) > 2 else 0.0,
                    "RG3": float(rg_probs[3]) if len(rg_probs) > 3 else 0.0,
                    "PDR": float(rg_probs[4]) if len(rg_probs) > 4 else 0.0
                },
                "dme_status": ["none", "non_center_involving", "center_involving"][me_pred]
            },
            
            "grading_rules": {
                "hemorrhages_4_quadrants": hemorrhages_4q,
                "venous_beading_2plus_quadrants": venous_beading_2q,
                "irma_1plus_quadrant": irma_1q,
                "meets_4_2_1_rule": meets_421
            },
            
            "pdr_findings": {
                "NVD_area": ["none", "<1/3DD", "1/3-1DD", ">1DD"][
                    torch.argmax(outputs['nvd_area_logits'][idx]).item()
                ] if 'nvd_area_logits' in outputs else "none",
                "NVE_area": ["none", "<1/2DD", "≥1/2DD"][
                    torch.argmax(outputs['nve_area_logits'][idx]).item()
                ] if 'nve_area_logits' in outputs else "none",
                "NV_activity": ["active", "fibrosed/regressed", "unknown"][
                    torch.argmax(outputs['nv_activity_logits'][idx]).item()
                ] if 'nv_activity_logits' in outputs else "unknown"
            },
            
            "image_quality": {
                "iq_score_0_4": outputs['image_quality_score'][idx].item() if 'image_quality_score' in outputs else 3.0
            }
        }
        
        return result
    
    def _determine_referable_dr(self, rg_grade: int, me_grade: int) -> bool:
        """Determine if DR is referable (≥ moderate NPDR or any DME)."""
        return rg_grade >= 2 or me_grade >= 1
    
    def _determine_sight_threatening(self, rg_grade: int, me_grade: int, pdr_activity: int) -> bool:
        """Determine if DR is sight-threatening (PDR or center-involving DME)."""
        return rg_grade == 4 or me_grade == 2 or pdr_activity > 0
    
    def _get_severity_name(self, grade: int) -> str:
        """Get severity name from grade."""
        names = {
            0: "no_DR",
            1: "mild_NPDR", 
            2: "moderate_NPDR",
            3: "severe_NPDR",
            4: "PDR"
        }
        return names.get(grade, "unknown")

class DiabeticRetinopathyModel(nn.Module):
    """Complete model for diabetic retinopathy classification with medical reasoning."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Vision backbone
        self.backbone = RETFoundBackbone(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            pretrained_path=config.model.pretrained_path
        )
        
        # Enhanced multi-task classification head
        self.classifier = MultiTaskHead(
            embed_dim=config.model.embed_dim,
            num_classes_rg=config.model.num_classes_rg,
            num_classes_me=config.model.num_classes_me,
            dropout=config.model.dropout,
            enable_referable=getattr(config.model, 'enable_referable_classification', True),
            enable_confidence=getattr(config.model, 'enable_confidence_estimation', True),
            enable_localization=getattr(config.model, 'enable_feature_localization', True)
        )
        
        # Medical reasoning module
        self.reasoning_module = MedicalReasoningModule(
            visual_dim=512,  # From language_features
            hidden_dim=768,
            max_length=config.language.max_length
        )
        
        # Clinical rule engine
        self.rule_engine = ClinicalRuleEngine(
            medical_terms_path=config.language.medical_terms_path
        )
        
        # Initialize tokenizer for text processing
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def forward(self, 
                images: torch.Tensor,
                text_input_ids: Optional[torch.Tensor] = None,
                text_attention_mask: Optional[torch.Tensor] = None,
                text_labels: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """
        Enhanced forward pass through complete model.
        
        Args:
            images: [batch_size, 3, H, W]
            text_input_ids: [batch_size, seq_len] - for training reasoning
            text_attention_mask: [batch_size, seq_len]
            text_labels: [batch_size, seq_len] - for reasoning loss
        """
        
        # Extract visual features
        visual_features = self.backbone(images)
        
        # Enhanced multi-task classification
        classification_outputs = self.classifier(visual_features)
        
        # Medical reasoning
        reasoning_outputs = self.reasoning_module(
            visual_features=classification_outputs['language_features'],
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            labels=text_labels
        )
        
        # Apply clinical rules (only during evaluation/inference, not training)
        clinical_outputs = []
        if not self.training:
            clinical_outputs = self.rule_engine.apply_clinical_rules(classification_outputs)
        
        return {
            **classification_outputs,
            'reasoning_loss': reasoning_outputs.get('loss'),
            'reasoning_logits': reasoning_outputs.get('logits'),
            'generated_text': reasoning_outputs.get('generated_ids'),
            'clinical_structured_output': clinical_outputs
        }
    
    def generate_comprehensive_report(self, 
                                    images: torch.Tensor,
                                    include_evidence: bool = True) -> List[Dict[str, any]]:
        """Generate comprehensive clinical reports."""
        
        with torch.no_grad():
            outputs = self.forward(images)
            
            reports = []
            for i, clinical_output in enumerate(outputs['clinical_structured_output']):
                
                # Generate reasoning text
                reasoning_outputs = self.reasoning_module.generate_medical_text(
                    outputs['language_features'][i:i+1],
                    max_new_tokens=100,
                    temperature=0.7
                )
                
                # Create comprehensive report
                report = {
                    **clinical_output,
                    "generated_clinical_narrative": self._decode_reasoning_text(reasoning_outputs),
                    "evidence": self._extract_evidence(outputs, i) if include_evidence else None,
                    "recommendations": self._generate_recommendations(clinical_output)
                }
                
                reports.append(report)
            
            return reports
    
    def generate_medical_report(self, 
                              images: torch.Tensor,
                              rg_predictions: torch.Tensor,
                              me_predictions: torch.Tensor) -> List[str]:
        """Generate medical reports for given images and predictions (legacy compatibility)."""
        
        comprehensive_reports = self.generate_comprehensive_report(images, include_evidence=False)
        
        # Convert to legacy format
        reports = []
        for i, comp_report in enumerate(comprehensive_reports):
            rg_grade = rg_predictions[i].item() if i < len(rg_predictions) else 0
            me_grade = me_predictions[i].item() if i < len(me_predictions) else 0
            
            narrative = comp_report.get("generated_clinical_narrative", "Unable to generate detailed analysis.")
            report = self._format_medical_report(rg_grade, me_grade, narrative)
            reports.append(report)
        
        return reports
    
    def _decode_reasoning_text(self, reasoning_outputs: Dict[str, torch.Tensor]) -> str:
        """Decode generated reasoning text."""
        if reasoning_outputs['generated_ids'].shape[1] > 0:
            return self.tokenizer.decode(
                reasoning_outputs['generated_ids'][0],
                skip_special_tokens=True
            )
        return "Unable to generate detailed clinical narrative."
    
    def _extract_evidence(self, outputs: Dict[str, torch.Tensor], idx: int) -> Dict[str, any]:
        """Extract evidence and localization information."""
        evidence = {}
        
        if 'spatial_attention' in outputs:
            evidence['attention_weights'] = outputs['spatial_attention'][idx].cpu().numpy().tolist()
            
        if 'lesion_bbox' in outputs:
            evidence['detected_lesions'] = outputs['lesion_bbox'][idx].cpu().numpy().tolist()
            
        return evidence
    
    def _generate_recommendations(self, clinical_output: Dict[str, any]) -> List[str]:
        """Generate clinical recommendations based on findings."""
        recommendations = []
        
        grading = clinical_output['grading']
        
        if grading['sight_threatening_DR']:
            recommendations.extend([
                "urgent_vitreoretinal_consult",
                "immediate_ophthalmology_referral"
            ])
        elif grading['referable_DR']:
            recommendations.append("immediate_ophthalmology_referral")
        else:
            recommendations.append("annual_screening")
            
        # Add specific recommendations based on findings
        if grading['class_probabilities']['PDR'] > 0.3:
            recommendations.append("anti_VEGF_evaluation")
            
        if clinical_output['grading_rules']['meets_4_2_1_rule']:
            recommendations.append("panretinal_photocoagulation")
            
        return recommendations
    
    def _format_medical_report(self, rg_grade: int, me_grade: int, generated_text: str) -> str:
        """Format medical report with structured information."""
        
        rg_descriptions = {
            0: "No diabetic retinopathy detected",
            1: "Mild non-proliferative diabetic retinopathy",
            2: "Moderate non-proliferative diabetic retinopathy", 
            3: "Severe non-proliferative diabetic retinopathy",
            4: "Proliferative diabetic retinopathy"
        }
        
        me_descriptions = {
            0: "No macular edema risk",
            1: "Low macular edema risk",
            2: "High macular edema risk"
        }
        
        report = f"""
DIABETIC RETINOPATHY ANALYSIS REPORT
====================================

RETINOPATHY GRADE: {rg_grade} - {rg_descriptions.get(rg_grade, 'Unknown')}
MACULAR EDEMA RISK: {me_grade} - {me_descriptions.get(me_grade, 'Unknown')}

CLINICAL FINDINGS:
{generated_text}

RECOMMENDATION:
{"Routine follow-up recommended" if rg_grade <= 1 and me_grade <= 1 
 else "Ophthalmological referral recommended for further evaluation"}
        """.strip()
        
        return report