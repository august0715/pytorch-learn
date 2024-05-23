import torch
from torch import nn

# https://blog.csdn.net/raelum/article/details/125462028
corpus = ["he is an old worker", "english is a useful tool", "the cinema is far away"]

def tokenize(corpus):
    return [sentence.split() for sentence in corpus]


tokens = tokenize(corpus)
print(tokens)

# [['he', 'is', 'an', 'old', 'worker'], ['english', 'is', 'a', 'useful', 'tool'], ['the', 'cinema', 'is', 'far', 'away']]

vocab = set(sum(tokens, []))
print(vocab)

from collections import Counter

word2idx = {
    word: idx
    for idx, (word, freq) in enumerate(
        sorted(Counter(sum(tokens, [])).items(), key=lambda x: x[1], reverse=True))
}
print(word2idx)

# {'is': 0, 'he': 1, 'an': 2, 'old': 3, 'worker': 4, 'english': 5, 'a': 6, 'useful': 7, 'tool': 8, 'the': 9, 'cinema': 10, 'far': 11, 'away': 12}
idx2word = {idx: word for word, idx in word2idx.items()}
print(idx2word)
# {0: 'is', 1: 'he', 2: 'an', 3: 'old', 4: 'worker', 5: 'english', 6: 'a', 7: 'useful', 8: 'tool', 9: 'the', 10: 'cinema', 11: 'far', 12: 'away'}

encoded_tokens = [[word2idx[token] for token in line] for line in tokens]
print(encoded_tokens)

# nn.Embedding(num_embeddings, embedding_dim)
# 其中 num_embeddings 是词表的大小，即 len(vocab)；embedding_dim 是词向量的维度。


torch.manual_seed(0)  # 为了复现性
emb = nn.Embedding(13, 3) #13是因为上面一个有13个词语

print(emb.weight)
sentence = torch.tensor(encoded_tokens[0])  # 一共有三个句子，这里只使用第一个句子
print(sentence)
# tensor([1, 0, 2, 3, 4])
print(emb(sentence))
# tensor([[-0.4339,  0.8487,  0.6920],
#         [-1.1258, -1.1524, -0.2506],
#         [-0.3160, -2.1152,  0.3223],
#         [-1.2633,  0.3500,  0.3081],
#         [ 0.1198,  1.2377,  1.1168]], grad_fn=<EmbeddingBackward0>)
print(emb.weight[sentence] == emb(sentence))
# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True],
#         [True, True, True],
#         [True, True, True]])