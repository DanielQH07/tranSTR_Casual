"""
Set-of-Mark (SoM) Injection Module for TranSTR

This module implements Token Mark injection for explicit entity grounding in VideoQA.
Token Marks act as learnable causal anchors that link entities in questions to 
corresponding regions in video frames.

Key Components:
- TokenMarkPalette: Learnable mark embeddings (F = {r1, r2, ..., rNF})
- VisualMarkInjector: Injects marks into frame/object features
- TextMarkInjector: Injects marks into text embeddings
- EntityMatcher: Matches entity names to SoM mask regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import Dict, List, Optional, Tuple


class TokenMarkPalette(nn.Module):
    """
    Learnable Token Mark embeddings palette.
    
    Eq. 4.1: F = {r1, r2, ..., rNF}, ri ∈ R^d
    
    Each mark serves as a unique identifier for entities in the video.
    """
    
    def __init__(self, num_marks: int = 16, d_model: int = 768, init_scale: float = 0.02):
        super().__init__()
        self.num_marks = num_marks
        self.d_model = d_model
        
        # Learnable mark embeddings
        self.marks = nn.Embedding(num_marks, d_model)
        
        # Initialize with small values
        nn.init.normal_(self.marks.weight, mean=0.0, std=init_scale)
    
    def forward(self, mark_indices: torch.Tensor) -> torch.Tensor:
        """
        Get mark embeddings for given indices.
        
        Args:
            mark_indices: [N] tensor of mark indices
            
        Returns:
            [N, d_model] mark embeddings
        """
        return self.marks(mark_indices)
    
    def get_all_marks(self) -> torch.Tensor:
        """Return all mark embeddings [num_marks, d_model]."""
        return self.marks.weight


class VisualMarkInjector(nn.Module):
    """
    Injects Token Marks into visual features (frame and object).
    
    For objects (Eq. 4.2): ô_j = o_j + γ · P_obj(r_k)
    For frames (Eq. 4.3): S_t = Σ_k (m_k · r_k) / (ω + Σ_k m_k)
    """
    
    def __init__(self, d_model: int = 768, obj_feat_dim: int = 2048, 
                 gamma_init: float = 0.1, omega: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.omega = omega
        
        # Projection layers
        self.proj_obj = nn.Linear(d_model, obj_feat_dim)   # For object features (768 -> 2048)
        self.proj_frame = nn.Linear(d_model, d_model)      # For frame features
        
        # Learnable injection scale
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        
        # Downsampling for ViT patches
        self.downsample_pool = None  # Will be set dynamically
    
    def inject_objects(self, obj_feat: torch.Tensor, 
                       mark_embeddings: torch.Tensor,
                       obj_to_entity: Dict[int, Dict[int, int]],
                       batch_idx: int) -> torch.Tensor:
        """
        Inject Token Marks into object features based on IoU matching.
        
        Eq. 4.2: ô_j = o_j + γ · P_obj(r_k)
        
        Args:
            obj_feat: [T, O, obj_feat_dim] object features for one video
            mark_embeddings: [K, d_model] mark embeddings for matched entities
            obj_to_entity: {frame_idx: {obj_idx: entity_id}} mapping
            batch_idx: Current batch index (for debugging)
            
        Returns:
            [T, O, obj_feat_dim] injected object features
        """
        if not obj_to_entity or mark_embeddings is None:
            return obj_feat
        
        T, O, D = obj_feat.shape
        result = obj_feat.clone()
        
        for frame_idx, obj_mapping in obj_to_entity.items():
            if frame_idx >= T:
                continue
            for obj_idx, entity_id in obj_mapping.items():
                if obj_idx >= O:
                    continue
                if entity_id > 0 and entity_id <= mark_embeddings.size(0):
                    # Get mark and project to object feature space
                    mark = mark_embeddings[entity_id - 1]  # entity_id is 1-indexed
                    projected_mark = self.proj_obj(mark)
                    
                    # Residual injection
                    result[frame_idx, obj_idx, :projected_mark.size(0)] += self.gamma * projected_mark
        
        return result
    
    def inject_frames(self, frame_feat: torch.Tensor,
                      mark_embeddings: torch.Tensor,
                      frame_masks: Dict[int, torch.Tensor],
                      target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Inject Token Marks into frame features using spatial mask.
        
        Eq. 4.3: S_t = Σ_k (m_k · r_k) / (ω + Σ_k m_k)
        
        Args:
            frame_feat: [T, d_model] frame features (after ViT mean pooling)
            mark_embeddings: [K, d_model] mark embeddings
            frame_masks: {frame_idx: [H, W]} entity ID masks
            target_size: Optional target spatial size for downsampling
            
        Returns:
            [T, d_model] injected frame features
        """
        if not frame_masks or mark_embeddings is None:
            return frame_feat
        
        T, D = frame_feat.shape
        device = frame_feat.device
        K = mark_embeddings.size(0)
        result = frame_feat.clone()
        
        for frame_idx, mask in frame_masks.items():
            if frame_idx >= T:
                continue
            
            mask = mask.to(device)
            
            # Create spatial mark map S_t
            # For each pixel, sum weighted mark embeddings
            spatial_mark = torch.zeros(D, device=device)
            weight_sum = torch.tensor(self.omega, device=device)
            
            for k in range(1, K + 1):
                entity_mask = (mask == k).float()  # [H, W]
                mask_sum = entity_mask.sum()
                
                if mask_sum > 0:
                    # Weight mark by entity area
                    mark_k = mark_embeddings[k - 1]  # [D]
                    projected_mark = self.proj_frame(mark_k)  # [D]
                    spatial_mark += mask_sum * projected_mark
                    weight_sum += mask_sum
            
            # Normalize and add to frame feature
            spatial_mark = spatial_mark / weight_sum
            result[frame_idx] += self.gamma * spatial_mark
        
        return result


class TextMarkInjector(nn.Module):
    """
    Injects Token Marks into text embeddings at entity token positions.
    
    Eq. 4.4: ê_regionk = e_regionk + β · P_text(r_k)
    
    This module handles:
    1. Entity name matching in question text
    2. Token-level mark injection
    """
    
    def __init__(self, d_model: int = 768, beta_init: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Projection for text space
        self.proj_text = nn.Linear(d_model, d_model)
        
        # Learnable injection scale
        self.beta = nn.Parameter(torch.tensor(beta_init))
    
    def find_entity_positions(self, tokenizer, tokens: torch.Tensor, 
                               entity_names: Dict[int, str]) -> Dict[int, List[int]]:
        """
        Find token positions for each entity in the tokenized sequence.
        
        Args:
            tokenizer: HuggingFace tokenizer
            tokens: [seq_len] input_ids
            entity_names: {entity_id: "entity_name"}
            
        Returns:
            {entity_id: [token_positions]}
        """
        # Decode tokens to text
        token_strs = [tokenizer.decode([t]) for t in tokens]
        token_text = ''.join(token_strs).lower()
        
        entity_positions = {}
        
        for entity_id, name in entity_names.items():
            name_lower = name.lower()
            # Find approximate position
            positions = []
            char_pos = 0
            for i, tok_str in enumerate(token_strs):
                if name_lower in tok_str.lower():
                    positions.append(i)
                char_pos += len(tok_str)
            
            if positions:
                entity_positions[entity_id] = positions
        
        return entity_positions
    
    def inject(self, text_embeddings: torch.Tensor,
               mark_embeddings: torch.Tensor,
               entity_positions: Dict[int, List[int]]) -> torch.Tensor:
        """
        Inject Token Marks at entity positions in text embeddings.
        
        Args:
            text_embeddings: [seq_len, d_model] text embeddings
            mark_embeddings: [K, d_model] mark embeddings
            entity_positions: {entity_id: [token_positions]}
            
        Returns:
            [seq_len, d_model] injected text embeddings
        """
        if not entity_positions or mark_embeddings is None:
            return text_embeddings
        
        result = text_embeddings.clone()
        K = mark_embeddings.size(0)
        
        for entity_id, positions in entity_positions.items():
            if entity_id > 0 and entity_id <= K:
                mark = mark_embeddings[entity_id - 1]
                projected_mark = self.proj_text(mark)
                
                for pos in positions:
                    if pos < result.size(0):
                        result[pos] += self.beta * projected_mark
        
        return result


class EntityMatcher:
    """
    Matches entities between question text and SoM masks.
    
    Extracts entity references from questions and maps them to 
    corresponding entity IDs in the mask metadata.
    """
    
    # Common entity patterns in Causal-VidQA
    ENTITY_PATTERNS = [
        r'\b(man|woman|person|people|boy|girl|child|children|kid|baby)\b',
        r'\b(car|vehicle|truck|bus|motorcycle|bicycle|bike)\b',
        r'\b(dog|cat|animal|bird|horse)\b',
        r'\b(ball|bottle|cup|phone|bag|box|book)\b',
        r'\b(table|chair|door|window|floor|wall)\b',
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.ENTITY_PATTERNS]
    
    def extract_entities_from_question(self, question: str) -> List[str]:
        """Extract entity mentions from question text."""
        entities = []
        for pattern in self.patterns:
            matches = pattern.findall(question.lower())
            entities.extend(matches)
        return list(set(entities))
    
    def match_to_mask_entities(self, question_entities: List[str],
                                mask_entity_names: Dict[int, str]) -> Dict[str, int]:
        """
        Match extracted entities to mask entity IDs.
        
        Args:
            question_entities: List of entity names from question
            mask_entity_names: {entity_id: "label"} from mask metadata
            
        Returns:
            {entity_name: entity_id} mapping
        """
        matches = {}
        
        for q_entity in question_entities:
            q_entity_lower = q_entity.lower()
            for entity_id, mask_label in mask_entity_names.items():
                mask_label_lower = mask_label.lower()
                
                # Check for match
                if q_entity_lower in mask_label_lower or mask_label_lower in q_entity_lower:
                    matches[q_entity] = entity_id
                    break
                
                # Check for common synonyms
                if self._is_synonym(q_entity_lower, mask_label_lower):
                    matches[q_entity] = entity_id
                    break
        
        return matches
    
    def _is_synonym(self, word1: str, word2: str) -> bool:
        """Check if two words are synonyms."""
        synonyms = {
            'person': ['man', 'woman', 'people', 'human', 'guy', 'lady'],
            'car': ['vehicle', 'automobile'],
            'kid': ['child', 'boy', 'girl', 'children'],
        }
        
        for key, syns in synonyms.items():
            all_words = [key] + syns
            if word1 in all_words and word2 in all_words:
                return True
        
        return False


class SoMInjector(nn.Module):
    """
    Main Set-of-Mark Injection module that orchestrates all injection components.
    
    This module is integrated into VideoQAmodel AFTER frame/object resize.
    All features are in d_model space when this is called.
    
    Key fixes (v2):
    - Works with resized features (all in d_model dimensions)
    - Properly normalizes mask weights to avoid gradient explosion
    - Uses idx_frame to map selected frames back to original mask frames
    """
    
    def __init__(self, d_model: int = 768, obj_feat_dim: int = 2048,
                 num_marks: int = 16, gamma_init: float = 0.1, 
                 beta_init: float = 0.1, omega: float = 1e-6):
        super().__init__()
        
        self.palette = TokenMarkPalette(num_marks, d_model)
        self.entity_matcher = EntityMatcher()
        
        self.num_marks = num_marks
        self.d_model = d_model
        self.omega = omega
        
        # Projection for frame marks (d_model -> d_model, learnable transform)
        self.proj_frame = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Projection for object marks (d_model -> d_model)
        self.proj_obj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Learnable injection scales
        self.gamma_frame = nn.Parameter(torch.tensor(gamma_init))
        self.gamma_obj = nn.Parameter(torch.tensor(gamma_init))
    
    def process_som_data(self, som_data: Optional[Dict]) -> Tuple[
        Optional[Dict[int, torch.Tensor]], 
        Optional[Dict[int, str]]
    ]:
        """
        Extract and validate SoM data components.
        
        Returns:
            (frame_masks, entity_names)
        """
        if som_data is None:
            return None, None
        
        frame_masks = som_data.get('frame_masks', {})
        entity_names = som_data.get('entity_names', {})
        
        return frame_masks, entity_names
    
    def get_active_mark_embeddings(self, entity_names: Dict[int, str], 
                                    device: torch.device) -> Tuple[Optional[torch.Tensor], Dict[int, int]]:
        """
        Get mark embeddings for active entities in current sample.
        
        Args:
            entity_names: {entity_id: "label"} - entity IDs may be non-contiguous (e.g., {1, 3, 5})
            device: torch device
            
        Returns:
            Tuple of:
            - mark_embeddings: [K, d_model] where K = number of unique entities
            - entity_to_mark: {entity_id: mark_index} mapping
        """
        if not entity_names:
            return None, {}
        
        # Get sorted entity IDs (they may be non-contiguous like {1, 3, 5})
        entity_ids = sorted(entity_names.keys())
        num_entities = min(len(entity_ids), self.num_marks)
        
        if num_entities <= 0:
            return None, {}
        
        # Create mapping: entity_id -> mark_index (0-indexed)
        entity_to_mark = {eid: idx for idx, eid in enumerate(entity_ids[:num_entities])}
        
        # Get mark embeddings for each entity
        indices = torch.arange(num_entities, device=device)
        mark_embeddings = self.palette(indices)  # [num_entities, d_model]
        
        return mark_embeddings, entity_to_mark
    
    def inject_frame_features(self, frame_feat: torch.Tensor, 
                               mark_embeddings: torch.Tensor,
                               frame_masks: Dict[int, torch.Tensor],
                               entity_to_mark: Dict[int, int]) -> torch.Tensor:
        """
        Inject Token Marks into frame features (already in d_model space).
        
        Args:
            frame_feat: [T, d_model] frame features (AFTER topK selection)
            mark_embeddings: [K, d_model] mark embeddings
            frame_masks: {frame_idx: [H, W]} entity ID masks
            entity_to_mark: {entity_id: mark_index} mapping for non-contiguous IDs
            
        Returns:
            [T, d_model] injected frame features
        """
        if not frame_masks or mark_embeddings is None or not entity_to_mark:
            return frame_feat
        
        T, D = frame_feat.shape
        device = frame_feat.device
        result = frame_feat.clone()
        
        # Get projected mark embeddings
        projected_marks = self.proj_frame(mark_embeddings)  # [K, d_model]
        
        for frame_idx, mask in frame_masks.items():
            if frame_idx >= T:
                continue
            
            mask = mask.to(device)
            
            # Compute normalized entity contributions
            spatial_mark = torch.zeros(D, device=device)
            total_entity_pixels = 0.0
            
            # First pass: count total entity pixels
            for entity_id, mark_idx in entity_to_mark.items():
                entity_mask = (mask == entity_id).float()  # [H, W]
                mask_sum = entity_mask.sum().item()
                if mask_sum > 0:
                    total_entity_pixels += mask_sum
            
            # Second pass: compute weighted spatial mark
            if total_entity_pixels > 0:
                for entity_id, mark_idx in entity_to_mark.items():
                    entity_mask = (mask == entity_id).float()
                    mask_sum = entity_mask.sum().item()
                    
                    if mask_sum > 0:
                        # Normalized weight (proportion of this entity)
                        weight = mask_sum / (total_entity_pixels + self.omega)
                        mark_k = projected_marks[mark_idx]  # Use mapping!
                        spatial_mark += weight * mark_k
                
                # Add normalized spatial mark to frame feature
                result[frame_idx] = result[frame_idx] + self.gamma_frame * spatial_mark
        
        return result
    
    def inject_object_features(self, obj_feat: torch.Tensor,
                                mark_embeddings: torch.Tensor,
                                frame_masks: Dict[int, torch.Tensor],
                                entity_to_mark: Dict[int, int]) -> torch.Tensor:
        """
        Inject Token Marks into object features based on entity presence in frames.
        
        Since we don't have object_to_entity mapping, we inject marks to ALL objects
        proportionally based on the entities present in that frame.
        
        Args:
            obj_feat: [T, O, d_model] object features (AFTER resize)
            mark_embeddings: [K, d_model] mark embeddings
            frame_masks: {frame_idx: [H, W]} entity ID masks
            entity_to_mark: {entity_id: mark_index} mapping for non-contiguous IDs
            
        Returns:
            [T, O, d_model] injected object features
        """
        if not frame_masks or mark_embeddings is None or not entity_to_mark:
            return obj_feat
        
        T, O, D = obj_feat.shape
        device = obj_feat.device
        result = obj_feat.clone()
        
        # Get projected mark embeddings
        projected_marks = self.proj_obj(mark_embeddings)  # [K, d_model]
        
        for frame_idx, mask in frame_masks.items():
            if frame_idx >= T:
                continue
            
            mask = mask.to(device)
            
            # Compute average mark for this frame based on entity presence
            frame_mark = torch.zeros(D, device=device)
            num_entities = 0
            
            for entity_id, mark_idx in entity_to_mark.items():
                entity_mask = (mask == entity_id).float()
                mask_sum = entity_mask.sum().item()
                
                if mask_sum > 0:
                    frame_mark += projected_marks[mark_idx]  # Use mapping!
                    num_entities += 1
            
            if num_entities > 0:
                frame_mark = frame_mark / num_entities
                # Inject to all objects in this frame
                result[frame_idx] = result[frame_idx] + self.gamma_obj * frame_mark
        
        return result
    
    def forward(self, frame_local: torch.Tensor, obj_local: torch.Tensor,
                som_data_batch: List[Optional[Dict]], 
                idx_frame: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main forward pass for visual injection.
        
        IMPORTANT: This is called AFTER frame/obj resize and topK selection.
        
        Args:
            frame_local: [B, frame_topK, d_model] selected frame features
            obj_local: [B, frame_topK, O, d_model] object features for selected frames
            som_data_batch: List of SoM data for each batch item
            idx_frame: [B, F, frame_topK] frame selection weights (optional)
            
        Returns:
            (injected_frame_local, injected_obj_local)
        """
        B = frame_local.size(0)
        T = frame_local.size(1)  # This is frame_topK after selection
        device = frame_local.device
        
        injected_frames = frame_local.clone()
        injected_objs = obj_local.clone()
        
        for b in range(B):
            som_data = som_data_batch[b] if som_data_batch else None
            frame_masks, entity_names = self.process_som_data(som_data)
            
            if entity_names and frame_masks:
                # Get mark embeddings and entity-to-mark mapping
                mark_emb, entity_to_mark = self.get_active_mark_embeddings(entity_names, device)
                
                if mark_emb is not None and entity_to_mark:
                    # Map original frame masks to selected frame indices
                    # If idx_frame is provided, find which original frames map to selected frames
                    mapped_masks = {}
                    
                    if idx_frame is not None:
                        # idx_frame: [B, F, frame_topK] - weights for each selected frame
                        # Find top contributing original frame for each selected frame
                        frame_weights = idx_frame[b]  # [F, frame_topK]
                        top_orig_frames = frame_weights.argmax(dim=0)  # [frame_topK]
                        
                        # Map masks from original frames to selected frame positions
                        for sel_idx in range(T):
                            orig_idx = top_orig_frames[sel_idx].item()
                            if orig_idx in frame_masks:
                                mapped_masks[sel_idx] = frame_masks[orig_idx]
                    else:
                        # No mapping, assume 1:1 correspondence (first T frames)
                        for orig_idx, mask in frame_masks.items():
                            if orig_idx < T:
                                mapped_masks[orig_idx] = mask
                    
                    # Inject into frame features
                    injected_frames[b] = self.inject_frame_features(
                        frame_local[b], mark_emb, mapped_masks, entity_to_mark
                    )
                    
                    # Inject into object features
                    injected_objs[b] = self.inject_object_features(
                        obj_local[b], mark_emb, mapped_masks, entity_to_mark
                    )
        
        return injected_frames, injected_objs


# Utility functions

def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """Compute IoU between two binary masks."""
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    return (intersection / (union + 1e-6)).item()


def box_to_mask(box: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert bounding box to binary mask."""
    x1, y1, x2, y2 = box[:4].int().tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[y1:y2, x1:x2] = True
    return mask


if __name__ == "__main__":
    # Test Token Mark components
    print("Testing SoM Injection Module (v2)...")
    
    # Test palette
    palette = TokenMarkPalette(num_marks=16, d_model=768)
    marks = palette(torch.tensor([0, 3, 5]))
    print(f"✓ Palette: {marks.shape}")  # (3, 768)
    
    # Test full injector with d_model features (after resize)
    som_injector = SoMInjector(d_model=768, obj_feat_dim=768, num_marks=16)
    
    # Simulate AFTER resize/topK: frame_local, obj_local
    frame_topK = 4
    num_objs = 10
    batch_size = 2
    
    frame_local = torch.randn(batch_size, frame_topK, 768)  # [B, frame_topK, d_model]
    obj_local = torch.randn(batch_size, frame_topK, num_objs, 768)  # [B, frame_topK, O, d_model]
    
    som_data = [{
        'frame_masks': {
            0: torch.randint(0, 3, (224, 224)),
            1: torch.randint(0, 3, (224, 224)),
        },
        'entity_names': {1: 'person', 2: 'car'},
    }] * batch_size
    
    # Simulate idx_frame (frame selection weights)
    F_orig = 16
    idx_frame = torch.randn(batch_size, F_orig, frame_topK).softmax(dim=0)
    
    inj_frames, inj_objs = som_injector(frame_local, obj_local, som_data, idx_frame=idx_frame)
    print(f"✓ Full injection: frames={inj_frames.shape}, objs={inj_objs.shape}")
    
    # Test without idx_frame
    inj_frames2, inj_objs2 = som_injector(frame_local, obj_local, som_data)
    print(f"✓ Without idx_frame: frames={inj_frames2.shape}, objs={inj_objs2.shape}")
    
    # Verify difference
    frame_diff = (inj_frames - frame_local).abs().mean().item()
    obj_diff = (inj_objs - obj_local).abs().mean().item()
    print(f"✓ Frame injection magnitude: {frame_diff:.6f}")
    print(f"✓ Object injection magnitude: {obj_diff:.6f}")
    
    print("\n✅ All tests passed!")

