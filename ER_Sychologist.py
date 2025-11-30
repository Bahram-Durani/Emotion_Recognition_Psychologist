# ADD THESE LINES AT THE VERY TOP OF YOUR SCRIPT (BEFORE ANY OTHER IMPORTS)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Now continue with the rest of your imports
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from tqdm import tqdm
from einops import rearrange
import json
import logging
from datetime import datetime
import sys  # Added to fix progress bar issue
from torch.amp import GradScaler, autocast  # For mixed precision

# ================================
# 1. Configure logging
# ================================
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("="*50)
logging.info("Starting E-Branchformer Trimodal Emotion Recognition Training")
logging.info("="*50)

# ================================
# 2. E-Branchformer Implementation
# ================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with large kernel size (31) as in the paper"""
    def __init__(self, channels, kernel_size=31):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, 
            groups=channels, padding=kernel_size//2
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # Input: (batch, seq_len, channels)
        x = rearrange(x, 'b s c -> b c s')
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = rearrange(x, 'b c s -> b s c')
        return self.norm(x)

class CGMLP(nn.Module):
    """Convolution-gated MLP module from the paper"""
    def __init__(self, d_model, d_ffn, dropout=0.25):  # CHANGED FROM 0.1 TO 0.25
        super().__init__()
        self.conv = DepthwiseSeparableConv(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.swish = nn.SiLU()  # Swish activation as in the paper
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x)
        gate = torch.sigmoid(self.gate_proj(x))
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x * gate + residual

class SelfAttention(nn.Module):
    """Self-attention module for the E-Branchformer"""
    def __init__(self, d_model, n_heads, dropout=0.25):  # CHANGED FROM 0.1 TO 0.25
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)
        
        # Project to query, key, value
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose for attention dot product
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq_len, d_head)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, -1)
        
        # Final projection
        out = self.out_proj(out)
        return out + residual

class MergeModule(nn.Module):
    """Merge module to combine features from different branches"""
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, attn_out, cgmlp_out):
        out = torch.cat([attn_out, cgmlp_out], dim=-1)
        out = self.fc(out)
        return self.norm(out)

class EBranchformerLayer(nn.Module):
    """Single E-Branchformer layer as described in the paper"""
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.25):  # CHANGED FROM 0.1 TO 0.25
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.cgmlp = CGMLP(d_model, d_ffn, dropout)
        self.merge = MergeModule(d_model)
        
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        cgmlp_out = self.cgmlp(x)
        return self.merge(attn_out, cgmlp_out)

class EBranchformerEncoder(nn.Module):
    """E-Branchformer encoder with multiple layers"""
    def __init__(self, d_model, n_heads, d_ffn, num_layers, dropout=0.25):  # CHANGED FROM 0.1 TO 0.25
        super().__init__()
        self.layers = nn.ModuleList([
            EBranchformerLayer(d_model, n_heads, d_ffn, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EBranchformer(nn.Module):
    """Complete E-Branchformer network matching paper specifications"""
    def __init__(self, input_dim, output_dim=50, n_heads=8, d_ffn=2048, num_layers=1, dropout=0.25):  # CHANGED FROM 0.1 TO 0.25
        """
        input_dim: Input feature dimension
        output_dim: Output feature dimension (50 as in paper)
        n_heads: Number of attention heads
        d_ffn: Feed-forward network dimension (2048 as in paper)
        num_layers: Number of E-Branchformer layers
        dropout: Dropout rate
        """
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_ffn)
        
        # E-Branchformer encoder
        self.encoder = EBranchformerEncoder(
            d_model=d_ffn,
            n_heads=n_heads,
            d_ffn=d_ffn,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_ffn, output_dim)
        self.swish = nn.SiLU()
        
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch, seq_len, output_dim)
        """
        x = self.input_proj(x)
        x = self.swish(x)
        x = self.encoder(x)
        return self.output_proj(x)

# ================================
# 3. Trimodal Emotion Recognition Model
# ================================

class TrimodalEmotionModel(nn.Module):
    """Complete trimodal emotion recognition model based on E-Branchformer"""
    def __init__(self, video_dim=1408, audio_dim=1024, pose_dim=156, num_classes=6, dropout=0.25):  # CHANGED FROM 0.1 TO 0.25
        super().__init__()
        
        # Modal Processing Layer - One E-Branchformer per modality (Section 4.3.2)
        self.visual_layer = EBranchformer(
            input_dim=video_dim, 
            output_dim=50,  # Unified 50-dimensional space (Section 3.1)
            n_heads=8,
            num_layers=1,  # One E-Branchformer as per optimal config
            dropout=dropout
        )
        
        self.acoustic_layer = EBranchformer(
            input_dim=audio_dim,
            output_dim=50,
            n_heads=8,
            num_layers=1,
            dropout=dropout
        )
        
        self.pose_layer = EBranchformer(
            input_dim=pose_dim,
            output_dim=50,
            n_heads=8,
            num_layers=1,
            dropout=dropout
        )
        
        # Shared Layer - Two E-Branchformers (Section 4.3.2)
        self.shared_layer = EBranchformer(
            input_dim=150,  # 50*3 modalities (Section 3.1)
            output_dim=512,  # Intermediate representation
            n_heads=8,
            num_layers=2,  # Two layers as per optimal config
            dropout=dropout
        )
        
        # CORRECTED FEATURE PROJECTION (512 → 1536)
        self.feature_proj = nn.Sequential(
            nn.Linear(512, 1536),  # Paper specifies 512 → 1536
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Emotion Classification Layer (Section 3.1)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Reconstruction heads (Section 3.3)
        self.recon_proj = nn.Linear(512, 50)  # For reconstruction loss
        
        # Reconstruction target projection (2588 → 50)
        # 2588 = 1408 (video) + 1024 (audio) + 156 (pose)
        self.recon_target_proj = nn.Linear(2588, 50)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def temporal_pooling(self, x):
        """Mean pooling across time dimension (Sec 3.3)"""
        return x.mean(dim=1)
    
    def forward(self, video, audio, pose, labels=None):
        """
        video: (batch, 90, 1408) 
        audio: (batch, 150, 1024)
        pose: (batch, 90, 156)
        """
        # Process each modality (Section 3.1)
        visual_features = self.visual_layer(video)  # (B, 90, 50)
        acoustic_features = self.acoustic_layer(audio)  # (B, 150, 50)
        pose_features = self.pose_layer(pose)  # (B, 90, 50)
        
        # Temporal pooling (Section 3.3)
        visual_pooled = self.temporal_pooling(visual_features)  # (B, 50)
        acoustic_pooled = self.temporal_pooling(acoustic_features)  # (B, 50)
        pose_pooled = self.temporal_pooling(pose_features)  # (B, 50)
        
        # Concatenate for fusion (Section 3.1)
        fused = torch.cat([
            visual_pooled, 
            acoustic_pooled, 
            pose_pooled
        ], dim=1)  # (B, 150)
        
        # Shared Layer processing (Section 3.1)
        shared_out = self.shared_layer(fused.unsqueeze(1))  # (B, 1, 512)
        shared_pooled = shared_out.squeeze(1)  # (B, 512)
        
        # Project to 1536 dimensions (Paper Section 3.1)
        projected_features = self.feature_proj(shared_pooled)
        
        # Classification (Section 3.1)
        logits = self.classifier(projected_features)
        
        # Reconstruction (Section 3.3)
        rec_features = self.recon_proj(shared_pooled)  # (B, 50)
        
        # ADD NORMALIZATION FOR RECONSTRUCTION (CRITICAL FIX)
        rec_features = F.normalize(rec_features, p=2, dim=1)
        
        # Reconstruction target from RAW features (paper specification)
        video_pool = video.mean(dim=1)  # [B, 1408]
        audio_pool = audio.mean(dim=1)  # [B, 1024]
        pose_pool = pose.mean(dim=1)    # [B, 156]
        x = torch.cat([video_pool, audio_pool, pose_pool], dim=1)  # [B, 2588]
        x_pool = self.recon_target_proj(x)  # [B, 50]
        
        # ADD NORMALIZATION FOR RECONSTRUCTION (CRITICAL FIX)
        x_pool = F.normalize(x_pool, p=2, dim=1)
        
        # Loss calculation (Section 3.3)
        loss = None
        if labels is not None:
            # Classification loss
            cls_loss = F.cross_entropy(logits, labels)
            
            # Reconstruction loss (using normalized features)
            recon_loss = F.mse_loss(rec_features, x_pool)
            
            # CRITICAL FIX: Increase reconstruction weight to 0.4
            loss = 0.6 * cls_loss + 0.4 * recon_loss  # Was 0.7/0.3
            
            # Debugging NaN losses
            if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                cls_loss = torch.tensor(0.0, device=cls_loss.device)
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(0.0, device=recon_loss.device)
        
        return {
            "logits": logits,
            "loss": loss,
            "cls_loss": 0.6 * cls_loss if labels is not None else None,
            "recon_loss": 0.4 * recon_loss if labels is not None else None,
            "rec_features": rec_features
        }

# ================================
# 4. Dataset and DataLoader
# ================================

class TrimodalDataset(torch.utils.data.Dataset):
    """Dataset for trimodal emotion recognition with sequence length handling"""
    def __init__(self, feature_dir, max_video_len=90, max_audio_len=150, 
                 video_mean=None, video_std=None, 
                 audio_mean=None, audio_std=None,
                 pose_mean=None, pose_std=None):
        self.feature_dir = feature_dir
        self.max_video_len = max_video_len
        self.max_audio_len = max_audio_len
        
        # Find all samples using video files as reference
        self.samples = [
            f.replace("_video_frames.npy", "")
            for f in os.listdir(feature_dir)
            if f.endswith("_video_frames.npy")
        ]
        
        # Use precomputed statistics or calculate if not provided
        if video_mean is None or video_std is None:
            # Only calculate if not provided (for training set)
            self.video_mean, self.video_std = self._compute_stats("video")
        else:
            self.video_mean, self.video_std = video_mean, video_std
            
        if audio_mean is None or audio_std is None:
            self.audio_mean, self.audio_std = self._compute_stats("audio")
        else:
            self.audio_mean, self.audio_std = audio_mean, audio_std
            
        if pose_mean is None or pose_std is None:
            self.pose_mean, self.pose_std = self._compute_stats("pose")
        else:
            self.pose_mean, self.pose_std = pose_mean, pose_std
        
        logging.info(f"Found {len(self.samples)} samples in {feature_dir}")
        print(f"Found {len(self.samples)} samples in {feature_dir}")
    
    def _compute_stats(self, modality):
        """Compute mean and std across the entire dataset for a modality"""
        all_features = []
        
        # Sample up to 1000 samples for statistics (adjust as needed)
        for sample_id in self.samples[:min(1000, len(self.samples))]:
            try:
                if modality == "video":
                    feature_path = os.path.join(self.feature_dir, f"{sample_id}_video_frames.npy")
                elif modality == "audio":
                    feature_path = os.path.join(self.feature_dir, f"{sample_id}_audio_frames.npy")
                else:  # pose
                    feature_path = os.path.join(self.feature_dir, f"{sample_id}_pose.npy")
                
                if os.path.exists(feature_path):
                    feature = np.load(feature_path)
                    # Only use the first 100 frames for statistics
                    if feature.shape[0] > 100:
                        feature = feature[:100]
                    all_features.append(feature)
            except Exception as e:
                logging.warning(f"Error loading {modality} for {sample_id}: {str(e)}")
        
        if all_features:
            all_features = np.concatenate(all_features, axis=0)
            mean = np.mean(all_features, axis=0)
            std = np.std(all_features, axis=0)
            return mean, std + 1e-8
        else:
            # Fallback values based on feature dimensions
            if modality == "video":
                dim = 1408
            elif modality == "audio":
                dim = 1024
            else:
                dim = 156
            return np.zeros(dim), np.ones(dim)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load features with safe loading
        video = np.load(os.path.join(self.feature_dir, f"{sample_id}_video_frames.npy"))
        audio = np.load(os.path.join(self.feature_dir, f"{sample_id}_audio_frames.npy"))
        pose = np.load(os.path.join(self.feature_dir, f"{sample_id}_pose.npy"))
        label = np.load(os.path.join(self.feature_dir, f"{sample_id}_label.npy"))
        
        # CRITICAL FIX: Use precomputed statistics (efficient normalization)
        video = (video - self.video_mean) / self.video_std
        audio = (audio - self.audio_mean) / self.audio_std
        pose = (pose - self.pose_mean) / self.pose_std
        
        # Ensure correct sequence lengths
        video = self._adjust_sequence(video, self.max_video_len, 1408)
        audio = self._adjust_sequence(audio, self.max_audio_len, 1024)
        pose = self._adjust_sequence(pose, self.max_video_len, 156)  # Same length as video
        
        return {
            "video": torch.tensor(video, dtype=torch.float32),
            "audio": torch.tensor(audio, dtype=torch.float32),
            "pose": torch.tensor(pose, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long).squeeze(0)  # Remove extra dim
        }
    
    def _adjust_sequence(self, arr, target_length, feature_dim):
        """Ensure sequences have exact length (pad or truncate)"""
        current_length = arr.shape[0]
        
        if current_length > target_length:
            # Central truncation
            start = (current_length - target_length) // 2
            return arr[start:start+target_length]
        elif current_length < target_length:
            # Padding with zeros
            pad = np.zeros((target_length - current_length, feature_dim))
            return np.vstack([arr, pad])
        return arr

def get_data_loaders(split_dir, batch_size=8):
    """Create data loaders for a specific split directory with efficient normalization"""
    # First, compute stats on training set
    train_dir = os.path.join(split_dir, "train")
    
    # Create training set to compute statistics
    train_set_temp = TrimodalDataset(train_dir)
    
    # Create all datasets with the same normalization stats
    train_set = TrimodalDataset(
        os.path.join(split_dir, "train"),
        video_mean=train_set_temp.video_mean, 
        video_std=train_set_temp.video_std,
        audio_mean=train_set_temp.audio_mean, 
        audio_std=train_set_temp.audio_std,
        pose_mean=train_set_temp.pose_mean, 
        pose_std=train_set_temp.pose_std
    )
    val_set = TrimodalDataset(
        os.path.join(split_dir, "val"),
        video_mean=train_set_temp.video_mean, 
        video_std=train_set_temp.video_std,
        audio_mean=train_set_temp.audio_mean, 
        audio_std=train_set_temp.audio_std,
        pose_mean=train_set_temp.pose_mean, 
        pose_std=train_set_temp.pose_std
    )
    test_set = TrimodalDataset(
        os.path.join(split_dir, "test"),
        video_mean=train_set_temp.video_mean, 
        video_std=train_set_temp.video_std,
        audio_mean=train_set_temp.audio_mean, 
        audio_std=train_set_temp.audio_std,
        pose_mean=train_set_temp.pose_mean, 
        pose_std=train_set_temp.pose_std
    )
    
    loaders = {
        "train": torch.utils.data.DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Required for Windows compatibility
            pin_memory=True,
            drop_last=True  # Important for batch norm
        ),
        "val": torch.utils.data.DataLoader(
            val_set, 
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        ),
        "test": torch.utils.data.DataLoader(
            test_set, 
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )
    }
    
    logging.info(f"Data loaded: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"Data loaded: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    # Log normalization statistics for verification
    logging.info("Normalization statistics:")
    logging.info(f"Video: mean={np.mean(train_set_temp.video_mean):.4f}, std={np.mean(train_set_temp.video_std):.4f}")
    logging.info(f"Audio: mean={np.mean(train_set_temp.audio_mean):.4f}, std={np.mean(train_set_temp.audio_std):.4f}")
    logging.info(f"Pose: mean={np.mean(train_set_temp.pose_mean):.4f}, std={np.mean(train_set_temp.pose_std):.4f}")
    
    return loaders

# ================================
# 5. Training and Evaluation Functions
# ================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and display confusion matrix with improved visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots - absolute and normalized
    plt.figure(figsize=(16, 7))
    
    # Absolute counts
    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Absolute Counts)')
    
    # Normalized
    plt.subplot(1, 2, 2)
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to {save_path}")
    
    # Display the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close()

def plot_modality_comparison(modality_results, save_dir, split_num):
    """
    Plot comparison of different modality combinations
    
    Args:
        modality_results: Dictionary of modality results
        save_dir: Directory to save the plot
        split_num: Current split number
    """
    # Collect results
    modalities = list(modality_results.keys())
    f1_macro = [modality_results[m]['f1_macro'] for m in modalities]
    f1_micro = [modality_results[m]['f1_micro'] for m in modalities]
    accuracy = [modality_results[m]['accuracy'] for m in modalities]
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # F1-Macro plot
    plt.subplot(1, 2, 1)
    x = np.arange(len(modalities))
    width = 0.25
    plt.bar(x - width, f1_macro, width, label='F1-Macro')
    plt.bar(x, f1_micro, width, label='F1-Micro')
    plt.bar(x + width, accuracy, width, label='Accuracy')
    plt.xlabel('Modality Combination')
    plt.ylabel('Score')
    plt.title(f'Split {split_num} - Modality Performance Comparison')
    plt.xticks(x, modalities)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(f1_macro):
        plt.text(i - width, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(f1_micro):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(accuracy):
        plt.text(i + width, v + 0.01, f'{v:.3f}', ha='center')
    
    # F1-Macro breakdown by emotion
    plt.subplot(1, 2, 2)
    class_names = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    full_results = modality_results['trimodal']
    
    # Get per-class F1 scores
    full_preds = full_results['preds']
    full_labels = full_results['labels']
    
    # Calculate per-class F1
    per_class_f1 = f1_score(full_labels, full_preds, average=None)
    
    # Plot per-class F1
    plt.bar(class_names, per_class_f1, color='skyblue')
    plt.xlabel('Emotion Class')
    plt.ylabel('F1 Score')
    plt.title('Per-Class Performance (Full Model)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(per_class_f1):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # Save and display
    plot_path = os.path.join(save_dir, f"modality_comparison_split{split_num}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    logging.info(f"Modality comparison plot saved to {plot_path}")
    
    plt.show()
    plt.close()
    
    return plot_path

def train_epoch(model, train_loader, optimizer, device, max_grad_norm=0.3):  # CHANGED FROM 0.5 TO 0.3
    """Train for one epoch with gradient clipping, NaN prevention, and gradient accumulation"""
    model.train()
    total_loss = 0
    total_samples = 0
    scaler = GradScaler(device="cuda")
    
    optimizer.zero_grad()
    backward_performed = False
    accumulation_steps = 2  # Number of steps to accumulate
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training", file=sys.stdout)):
        # Move data to device
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        pose = batch["pose"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass with mixed precision
        with autocast(device_type="cuda"):
            outputs = model(video, audio, pose, labels)
            loss = outputs["loss"]
        
        # NaN/Inf prevention
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            continue
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        backward_performed = True
        
        # Track loss
        total_loss += loss.item() * video.size(0)
        total_samples += video.size(0)
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            if backward_performed:
                # Unscaled for gradient clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping with tighter norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Step optimizer
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                backward_performed = False
    
    # Handle remaining gradients
    if backward_performed:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / total_samples if total_samples > 0 else float('nan')

def evaluate(model, data_loader, device, split_name="Validation", modalities=None):
    """Evaluate model with reconstruction loss tracking"""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_recon_loss = 0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    # For modality-specific evaluation
    modality_results = {}
    if modalities:
        for modality, mask_pattern in modalities.items():
            modality_results[modality] = {
                "loss": 0,
                "cls_loss": 0,
                "recon_loss": 0,
                "accuracy": 0,
                "f1_macro": 0,
                "f1_micro": 0,
                "preds": [],
                "labels": [],
                "count": 0
            }
    
    with torch.no_grad():
        # FIX: Added file=sys.stdout to ensure progress bar displays correctly
        for batch in tqdm(data_loader, desc=split_name, file=sys.stdout):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            pose = batch["pose"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type="cuda"):
                outputs = model(video, audio, pose, labels)
            
            # Track losses
            total_loss += outputs["loss"].item() * video.size(0)
            total_cls_loss += outputs["cls_loss"].item() * video.size(0) if outputs["cls_loss"] is not None else 0
            total_recon_loss += outputs["recon_loss"].item() * video.size(0) if outputs["recon_loss"] is not None else 0
            total_samples += video.size(0)
            
            # Track predictions
            preds = torch.argmax(outputs["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Modality-specific evaluation
            if modalities:
                for modality, mask_pattern in modalities.items():
                    # CRITICAL FIX: Prevent zero inputs that break batch normalization
                    masked_video = video * mask_pattern[0] + 1e-6
                    masked_audio = audio * mask_pattern[1] + 1e-6
                    masked_pose = pose * mask_pattern[2] + 1e-6
                    
                    # Evaluate with masked inputs
                    with autocast(device_type="cuda"):
                        mod_outputs = model(masked_video, masked_audio, masked_pose, labels)
                    
                    # Track modality-specific metrics
                    modality_results[modality]["loss"] += mod_outputs["loss"].item() * video.size(0)
                    modality_results[modality]["cls_loss"] += mod_outputs["cls_loss"].item() * video.size(0) if mod_outputs["cls_loss"] is not None else 0
                    modality_results[modality]["recon_loss"] += mod_outputs["recon_loss"].item() * video.size(0) if mod_outputs["recon_loss"] is not None else 0
                    modality_results[modality]["count"] += video.size(0)
                    
                    # Track predictions
                    mod_preds = torch.argmax(mod_outputs["logits"], dim=1)
                    modality_results[modality]["preds"].extend(mod_preds.cpu().numpy())
                    modality_results[modality]["labels"].extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    # Calculate per-modality metrics
    for modality in modality_results.keys():
        count = modality_results[modality]["count"]
        if count > 0:
            # Calculate average losses
            modality_results[modality]["loss"] /= count
            modality_results[modality]["cls_loss"] /= count
            modality_results[modality]["recon_loss"] /= count
            
            # Calculate metrics
            mod_preds = modality_results[modality]["preds"]
            mod_labels = modality_results[modality]["labels"]
            
            if mod_preds and mod_labels:
                modality_results[modality]["accuracy"] = accuracy_score(mod_labels, mod_preds)
                modality_results[modality]["f1_macro"] = f1_score(mod_labels, mod_preds, average='macro')
                modality_results[modality]["f1_micro"] = f1_score(mod_labels, mod_preds, average='micro')
    
    return {
        "loss": total_loss / total_samples,
        "cls_loss": total_cls_loss / total_samples,
        "recon_loss": total_recon_loss / total_samples,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "preds": all_preds,
        "labels": all_labels,
        "modality_results": modality_results if modalities else None
    }

def train_and_evaluate_split(split_num, split_dir, output_dir, device, epochs=50):
    """Complete training loop with paper-specified hyperparameters"""
    logging.info(f"\n{'='*50}")
    logging.info(f"TRAINING SPLIT {split_num}")
    logging.info(f"{'='*50}")
    print(f"\n{'='*50}")
    print(f"TRAINING SPLIT {split_num}")
    print(f"{'='*50}")
    
    # Setup directories
    split_output_dir = os.path.join(output_dir, f"split_{split_num}")
    os.makedirs(split_output_dir, exist_ok=True)
    
    # Data loaders
    loaders = get_data_loaders(
        split_dir,
        batch_size=8
    )
    
    # Model and optimizer (paper-specified)
    model = TrimodalEmotionModel().to(device)
    # Class weighting (disabled as per paper - class imbalance is minor)
    optimizer = optim.Adam(
        model.parameters(),
        lr=5e-5,  # 5 × 10^-5
        weight_decay=5e-6,  # CHANGED FROM 5e-7 TO 5e-6
        betas=(0.95, 0.999)  # Exponential moving averages
    )
    
    # CRITICAL FIX: Add learning rate warmup
    total_steps = len(loaders["train"]) * epochs
    warmup_steps = 500
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training state
    best_val_f1 = 0.0  # Changed from loss to F1-Macro (CRITICAL FIX)
    best_val_loss = float('inf')  # For early stopping guardrail
    patience = 0
    max_patience = 2  # CHANGED FROM 7 TO 2
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1_macro': [],
        'val_f1_micro': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        
        # Skip evaluation if no valid batches were processed
        if train_loss is None or np.isnan(train_loss):
            logging.warning(f"Skipping evaluation for epoch {epoch+1} due to no valid batches")
            continue
        
        # Validate
        val_results = evaluate(model, loaders["val"], device)
        
        # Update learning rate (after each epoch)
        scheduler.step()
        
        # CRITICAL FIX: Use F1-Macro for early stopping (NOT loss)
        current_f1 = val_results["f1_macro"]
        val_loss = val_results["loss"]
        
        # Save best F1 model (priority)
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 
                      os.path.join(split_output_dir, "best_model.pt"))
            logging.info(f"New best model saved at epoch {epoch+1} (F1-Macro: {current_f1:.4f})")
        else:
            # Track loss for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Early stopping logic
            if (current_f1 <= best_val_f1) and (val_loss > best_val_loss + 0.005):  # CHANGED FROM 0.02 TO 0.005
                patience += 1
                if patience >= max_patience:
                    logging.info(f"Early stopping at epoch {epoch+1} (F1 plateaued at {best_val_f1:.4f})")
                    print(f"Early stopping at epoch {epoch+1} (F1 plateaued at {best_val_f1:.4f})")
                    break
            else:
                patience = 0
        
        # Logging
        epoch_time = time.time() - start_time
        log_msg = (f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s\n"
                   f"  Train Loss: {train_loss:.4f} | Val Loss: {val_results['loss']:.4f}\n"
                   f"  Cls Loss: {val_results['cls_loss']:.4f} | Recon Loss: {val_results['recon_loss']:.4f}\n"
                   f"  Val F1-Macro: {val_results['f1_macro']:.4f} | F1-Micro: {val_results['f1_micro']:.4f} | Accuracy: {val_results['accuracy']:.4f}\n"
                   f"  Best Val F1: {best_val_f1:.4f} | Patience: {patience}/{max_patience}\n"
                   f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        logging.info(log_msg)
        print(log_msg)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_f1_macro'].append(val_results['f1_macro'])
        history['val_f1_micro'].append(val_results['f1_micro'])
        history['val_accuracy'].append(val_results['accuracy'])
    
    # Final evaluation
    try:
        model.load_state_dict(torch.load(os.path.join(split_output_dir, "best_model.pt"), weights_only=True))
    except:
        # If no model was saved (all epochs had NaN losses), use the current model
        logging.warning("No best model found, using current model for evaluation")
    
    # CRITICAL FIX: Use pattern tuples instead of pre-created tensors
    modalities = {
        'visual': (1, 0, 0),       # (video, audio, pose) mask pattern
        'acoustic': (0, 1, 0),     # (video, audio, pose) mask pattern
        'pose': (0, 0, 1),         # (video, audio, pose) mask pattern
        'trimodal': (1, 1, 1)      # (video, audio, pose) mask pattern
    }
    test_results = evaluate(model, loaders["test"], device, "Test", modalities=modalities)
    
    # Save results
    results = {
        'f1_macro': test_results['f1_macro'],
        'f1_micro': test_results['f1_micro'],
        'accuracy': test_results['accuracy'],
        'cls_loss': test_results['cls_loss'],
        'recon_loss': test_results['recon_loss'],
        'preds': test_results['preds'],
        'labels': test_results['labels'],
        'history': history,
        'modality_results': test_results['modality_results']
    }
    
    np.save(os.path.join(split_output_dir, "test_results.npy"), results)
    
    # Save confusion matrix
    class_names = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    plot_confusion_matrix(
        test_results['labels'], 
        test_results['preds'], 
        class_names,
        save_path=os.path.join(split_output_dir, "confusion_matrix.png")
    )
    
    # Save modality comparison
    plot_modality_comparison(
        test_results['modality_results'], 
        split_output_dir,
        split_num
    )
    
    # Save modality-specific results
    modality_msg = "\nModality-Specific Performance:"
    for modality, results in test_results['modality_results'].items():
        modality_msg += (f"\n  {modality}:"
                         f"\n    Accuracy = {results['accuracy']:.4f}"
                         f"\n    F1-Macro = {results['f1_macro']:.4f}"
                         f"\n    F1-Micro = {results['f1_micro']:.4f}")
    
    logging.info(modality_msg)
    print(modality_msg)
    
    # Print classification report
    report = classification_report(
        test_results['labels'], 
        test_results['preds'],
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:")
    print(report)
    
    # Save model config
    config = {
        "video_dim": 1408,
        "audio_dim": 1024,
        "pose_dim": 156,
        "num_classes": 6,
        "dropout": 0.25,  # CHANGED FROM 0.1 TO 0.25
        "learning_rate": 5e-5,
        "weight_decay": 5e-6,  # CHANGED FROM 5e-7 TO 5e-6
        "betas": (0.95, 0.999),
        "batch_size": 8,
        "max_grad_norm": 0.3,  # CHANGED FROM 0.5 TO 0.3
        "max_patience": 2,  # CHANGED FROM 7 TO 2
        "epochs": epochs
    }
    
    with open(os.path.join(split_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Print final results
    result_msg = (f"\nSplit {split_num} Test Results:\n"
                  f"  Accuracy: {test_results['accuracy']:.4f}\n"
                  f"  F1-Macro: {test_results['f1_macro']:.4f}\n"
                  f"  F1-Micro: {test_results['f1_micro']:.4f}\n"
                  f"  Classification Loss: {test_results['cls_loss']:.4f}\n"
                  f"  Reconstruction Loss: {test_results['recon_loss']:.4f}")
    
    logging.info(result_msg)
    print(result_msg)
    
    return test_results['f1_macro'], test_results['f1_micro'], test_results['accuracy']

def run_full_evaluation(base_dir, output_dir, device):
    """5-fold cross-validation as per paper"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    logging.info("\n" + "="*50)
    logging.info("STARTING 5-FOLD CROSS-VALIDATION")
    logging.info("="*50)
    print("\n" + "="*50)
    print("STARTING 5-FOLD CROSS-VALIDATION")
    print("="*50)
    
    for split_num in range(1, 6):
        split_dir = os.path.join(base_dir, f"split_{split_num}")
        if not os.path.exists(split_dir):
            logging.error(f"Split directory {split_dir} does not exist. Skipping split {split_num}.")
            print(f"Split directory {split_dir} does not exist. Skipping split {split_num}.")
            continue
            
        try:
            f1_macro, f1_micro, accuracy = train_and_evaluate_split(
                split_num, split_dir, output_dir, device
            )
            results.append((f1_macro, f1_micro, accuracy))
        except Exception as e:
            logging.error(f"Error training split {split_num}: {str(e)}")
            print(f"Error training split {split_num}: {str(e)}")
            # Continue with next split
            continue
    
    if not results:
        logging.error("No results to aggregate. Check if split directories exist.")
        print("No results to aggregate. Check if split directories exist.")
        return None
    
    # Aggregate results
    f1_macros = [r[0] for r in results]
    f1_micros = [r[1] for r in results]
    accuracies = [r[2] for r in results]
    
    final_results = {
        'f1_macro_mean': np.mean(f1_macros),
        'f1_macro_std': np.std(f1_macros),
        'f1_micro_mean': np.mean(f1_micros),
        'f1_micro_std': np.std(f1_micros),
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'per_split': results,
        'f1_macros': f1_macros,
        'f1_micros': f1_micros,
        'accuracies': accuracies
    }
    
    # Save final results
    np.save(os.path.join(output_dir, "final_results.npy"), final_results)
    
    # Save results to JSON for easy reading
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json_results = {
            "f1_macro_mean": float(final_results['f1_macro_mean']),
            "f1_macro_std": float(final_results['f1_macro_std']),
            "f1_micro_mean": float(final_results['f1_micro_mean']),
            "f1_micro_std": float(final_results['f1_micro_std']),
            "accuracy_mean": float(final_results['accuracy_mean']),
            "accuracy_std": float(final_results['accuracy_std']),
            "per_split": [{"split": i+1, 
                          "f1_macro": float(results[i][0]), 
                          "f1_micro": float(results[i][1]),
                          "accuracy": float(results[i][2])} 
                          for i in range(len(results))]
        }
        json.dump(json_results, f, indent=4)
    
    # Print final results
    result_msg = ("\nFinal Results:\n" +
                  f"  Accuracy: {final_results['accuracy_mean']:.4f} ± {final_results['accuracy_std']:.4f}\n" +
                  f"  Macro-F1: {final_results['f1_macro_mean']:.4f} ± {final_results['f1_macro_std']:.4f}\n" +
                  f"  Micro-F1: {final_results['f1_micro_mean']:.4f} ± {final_results['f1_micro_std']:.4f}")
    
    logging.info(result_msg)
    print(result_msg)
    
    # Compare with paper results
    paper_macro = 0.814
    paper_micro = 0.853
    
    comparison_msg = (f"\nComparison with Paper:\n" +
                      f"  Your Accuracy: {final_results['accuracy_mean']:.4f} ± {final_results['accuracy_std']:.4f}\n" +
                      f"  Your F1-Macro: {final_results['f1_macro_mean']:.4f} ± {final_results['f1_macro_std']:.4f}\n" +
                      f"  Paper F1-Macro: {paper_macro:.4f}\n" +
                      f"  Difference: {final_results['f1_macro_mean'] - paper_macro:.4f}\n\n" +
                      f"  Your F1-Micro: {final_results['f1_micro_mean']:.4f} ± {final_results['f1_micro_std']:.4f}\n" +
                      f"  Paper F1-Micro: {paper_micro:.4f}\n" +
                      f"  Difference: {final_results['f1_micro_mean'] - paper_micro:.4f}")
    
    logging.info(comparison_msg)
    print(comparison_msg)
    
    return final_results

# ================================
# 6. Main Execution
# ================================

if __name__ == "__main__":
    # Configuration
    SPLIT_DIR = r"E:\Research_Datasets\processed_features_split2"
    OUTPUT_DIR = os.path.join(
        os.path.dirname(SPLIT_DIR), 
        f"trimodal_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    print(f"Using device: {device}")
    
    # Save configuration
    config = {
        "split_dir": SPLIT_DIR,
        "output_dir": OUTPUT_DIR,
        "device": str(device),
        "batch_size": 8,
        "dropout": 0.25,  # CHANGED FROM 0.1 TO 0.25
        "learning_rate": 5e-5,
        "weight_decay": 5e-6,  # CHANGED FROM 5e-7 TO 5e-6
        "betas": [0.95, 0.999],
        "max_grad_norm": 0.3,  # CHANGED FROM 0.5 TO 0.3
        "early_stopping_patience": 2  # CHANGED FROM 7 TO 2
    }
    
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    logging.info("\nConfiguration:")
    logging.info(json.dumps(config, indent=4))
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Run full evaluation
    logging.info("\nStarting full evaluation...")
    print("\nStarting full evaluation...")
    final_results = run_full_evaluation(SPLIT_DIR, OUTPUT_DIR, device)
    
    if final_results:
        logging.info("\n" + "="*50)
        logging.info("TRAINING COMPLETE!")
        logging.info("="*50)
        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
    else:
        logging.error("\nTraining failed to complete.")
        print("\nTraining failed to complete.")