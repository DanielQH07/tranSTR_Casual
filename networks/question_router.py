import torch
import torch.nn as nn
import torch.nn.functional as F


FAMILY_TO_ID = {
    "descriptive": 0,
    "explanatory": 1,
    "predictive": 2,
    "counterfactual": 3,
    "predictive_reason": 4,
    "counterfactual_reason": 5,
}


class QuestionFamilyRouter(nn.Module):
    def __init__(self, mode="rule", d_model=768, num_families=6):
        super().__init__()
        self.mode = mode
        self.num_families = num_families
        self.classifier = nn.Linear(d_model, num_families)

    def _rule_map_one(self, question_text=None, question_type=None):
        if question_type in FAMILY_TO_ID:
            return FAMILY_TO_ID[question_type]

        text = (question_text or "").lower()
        if "counterfactual" in text or "what if" in text or "would happen if" in text:
            return FAMILY_TO_ID["counterfactual"]
        if "why" in text:
            # If dataset does not provide explicit type, map generic why to explanatory.
            return FAMILY_TO_ID["explanatory"]
        if "going to" in text or "next" in text or "will" in text:
            return FAMILY_TO_ID["predictive"]
        return FAMILY_TO_ID["descriptive"]

    def _to_tensor_ids(self, ids, device=None):
        if isinstance(ids, torch.Tensor):
            return ids.long()
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, question_emb=None, question_text=None, question_type=None, return_logits=False):
        if self.mode == "learned":
            if question_emb is None:
                raise ValueError("question_emb is required when mode='learned'")
            logits = self.classifier(question_emb)
            pred_ids = torch.argmax(logits, dim=-1)
            if return_logits:
                return pred_ids, logits
            return pred_ids

        # rule mode
        if isinstance(question_text, list):
            n = len(question_text)
            qtypes = question_type if isinstance(question_type, list) else [question_type] * n
            ids = [self._rule_map_one(question_text=question_text[i], question_type=qtypes[i]) for i in range(n)]
            ids = self._to_tensor_ids(ids)
        else:
            ids = self._rule_map_one(question_text=question_text, question_type=question_type)
            ids = self._to_tensor_ids([ids])

        if return_logits:
            logits = F.one_hot(ids, num_classes=self.num_families).float()
            return ids, logits
        return ids
