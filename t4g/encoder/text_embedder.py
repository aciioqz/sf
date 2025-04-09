from typing import List, Optional

import torch
# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    # 远程加载
    # def __init__(self, device: Optional[torch.device] = None):
    #     self.model = SentenceTransformer(
    #         "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device
    #     )


    # 服务器路径 /home/caoziqi/software/sentence-transformers/average_word_embeddings_glove.6B.300d
    def __init__(self, device: Optional[torch.device] = None):
        # path = "../encoder/average_word_embeddings_glove.6B.300d"
        # path = "/Users/caoziqi/Downloads/code/average_word_embeddings_glove.6B.300d"
        path = "/home/caoziqi/software/sentence-transformers/average_word_embeddings_glove.6B.300d"
        path='/root/code/average_word_embeddings_glove.6B.300d'
        self.model = SentenceTransformer(path, device=device)


    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))

def mean_pooling(last_hidden_state: Tensor, attention_mask) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)


