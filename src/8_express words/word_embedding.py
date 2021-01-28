# 워드 임베딩은 단어를 벡터로 표현하는 것을 말한다.

# 원-핫 인코딩을 통해서 나온 벡터들은 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지는 다 0이다.
# 이렇게 벡터 또는 행렬의 값이 대부분 0으로 표현되는 방법을 회소 표현이라고 한다.
# 이런 회소 벡터의 문제점은 단어의 개수가 늘어나면 벡터의 차원이 한 없이 커진다는 점이다.

import torch

# 원-핫 벡터 생성
dog = torch.FloatTensor([1, 0, 0, 0, 0])
cat = torch.FloatTensor([0, 1, 0, 0, 0])
computer = torch.FloatTensor([0, 0, 1, 0, 0])
netbook = torch.FloatTensor([0, 0, 0, 1, 0])
book = torch.FloatTensor([0, 0, 0, 0, 1])

# 이러한 원-핫 벡터간 코사인 유사도
print(torch.cosine_similarity(dog, cat, dim=0))
print(torch.cosine_similarity(cat, computer, dim=0))
print(torch.cosine_similarity(computer, netbook, dim=0))
print(torch.cosine_similarity(netbook, book, dim=0))
'''
다 동일하게 떠버린다.
'''

# 밀집 표현
# 위와 같이 희소 표현과 반대되는 표현이 있다. 바로 밀집 표현 방법이다.
# 밀집 표현은 벡터의 차원을 단어 집합의 크기로 상정하지 않았다.
# 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춘다.
# 또한, 이 과정에서 더 이상 0과 1만 가지 값이 아니라 실수의 값을 가지게 된다.


# 예를 들어, 만 개의 단어가 있을 때 강아지란 단어를 표현하기 위해서는 위 와 같은 표현을 사용했다. (원 핫 인코딩)
# 하지만 밀집 표현을 사용하고 사용자가 밀집 표현의 차원을 128로 설정했다면, 모든 단어의 벡터 포현의 차원은 128로 바뀌면서 모든 값이 실수가 된다.

# 단어를 밀집 벡터(dense vector)의 형태로 표현하는 방법을 워드 임베딩(word embedding)이라고 한다.
# 그리고 이 밀집 벡터를 워드 임베딩 과정을 통해 나온 결과라고 하여 임베딩 벡터(embedding vector)라고도 한다.