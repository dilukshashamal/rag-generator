import math

from src.config import Settings
from src.models.schemas import Evidence


class LocalCrossEncoderReranker:
    """Score bounded overlapping passages, retaining the original source for citations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None

    def rerank(self, question: str, evidence: list[Evidence]) -> list[Evidence]:
        if not evidence:
            return []
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.settings.reranker_model, device=self.settings.model_device,
                                      trust_remote_code=False, max_length=512)
        import torch
        evidence = evidence[:self.settings.rerank_top_k]
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for index, item in enumerate(evidence):
            for passage in self._passages(item.content):
                pairs.append((question, passage))
                owners.append(index)
        scores = self.model.predict(pairs,
                                    activation_fn=torch.nn.Identity(), show_progress_bar=False)
        best = [-60.0] * len(evidence)
        for owner, score in zip(owners, scores, strict=True):
            best[owner] = max(best[owner], float(score))
        result = [item.model_copy(update={"score": 1 / (1 + math.exp(-max(-60, min(60, score))))})
                  for item, score in zip(evidence, best, strict=True)]
        return sorted(result, key=lambda item: item.score, reverse=True)

    def _passages(self, content: str) -> list[str]:
        """Use the model's own tokenization and spread capped windows across the source.

        Decoded windows are used only for relevance scoring. Generation and grounding
        always receive the original chunk, with its casing, line breaks and qualifiers.
        """
        tokenizer = self.model.tokenizer
        tokens = tokenizer.encode(content, add_special_tokens=False)
        size = self.settings.reranker_passage_tokens
        if len(tokens) <= size:
            return [content]
        stride = size - self.settings.reranker_passage_overlap
        starts = list(range(0, max(1, len(tokens) - size + stride), stride))
        limit = self.settings.reranker_max_passages
        if len(starts) > limit:
            # Spread the bounded budget across the source instead of truncating its end.
            starts = ([starts[0]] if limit == 1 else
                      [starts[i * (len(starts) - 1) // (limit - 1)] for i in range(limit)])
        return [tokenizer.decode(tokens[start:start + size]) for start in starts]
