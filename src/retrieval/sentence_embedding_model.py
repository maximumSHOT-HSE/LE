import torch.nn as nn
import torch


class AutoModelForSentenceEmbedding(nn.Module):
    
    def __init__(self, model, tokenizer, normalize=True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.normalize = normalize

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = self.do_normalize(embeddings)
        return embeddings

    def do_normalize(self, x, eps: float = 1e-9):
        l2_norm = torch.linalg.vector_norm(x, ord=2)
        return x / max(l2_norm, eps)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
