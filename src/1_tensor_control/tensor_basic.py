import numpy as np

'''
!! 기본적인 개념
1차원 = 벡터
2차원 = 행렬
3차원 = 텐서
4차원 부터는 우리는 3차원의 세상에서 살고 있기 때문에 4차원 이상부터는 머리로 생각하기 어렵다.

2차원 텐서
2차원 텐서를 행렬이라고 말한다.
|t| = (batch size, dim)
batch size = "행" / dim = "열"

3차원 텐서
3차원 텐서는 그냥 텐서라고 부른다.
|t| = (batch size, width, height)
batch size = "세로" / width = "가로" / height = "높이" (입체적인 부분)
'''

# 1차원 벡터 만들기
t = np.array([0., 1., 2., 3., 4., 5., 6.])

# 벡터의 차원과 크기를 출력
print("Rank of t:", t.ndim)
print("Shape of t:", t.shape)
'''
ndim은 몇 차원인지를 출력한다.
shape은 크기를 출력한다. (1 x 7)의 크기를 말한다.
'''

# numpy에서 각 벡터의 원소에 접근하는 방법 (일반적인 파이썬 리스트를 다루는 것과 매우 유사)
print(t[0], t[1], t[-1])
print(t[:2], t[3:])

# 2차원 행렬 만들기
t = np.array([[1., 2., 3.,], [4., 5., 6.,], [7., 8., 9]])


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import torch

# 1차원 벡터 만들기
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

# 텐서의 차원 확인하기
print(t.dim()) # Rank (차원)
print(t.shape) # t.size()도 가능

# 텐서 접근하기 (일반적인 파이썬 리스트 접근 및 numpy접근과 동일하다.)
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

# PyTorch로 2차원 행렬 만들기
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]])
print(t)
print(t.dim()) # Rank (차원)
print(t.shape) # t.size()도 가능

# 2차원 텐서 슬라이싱
print(t[1:3, 1]) # 첫 번째 차원에서의 슬라이싱, 두 번째 차원에서의 인덱스의 값들만 가져온다.
print(t[1:3, 1].size())

# 브로드캐스팅 이용하기
'''
두 행렬 A, B에서 덧셈은 두 행렬의 크기가 같아야하고, 곱은 A의 마지막 차원과 B의 첫 번째 차원이 일치해야한다.
그치만 딥 러닝을 수행하다보면 불가피하게 크기가 다른 행렬(텐서)에 대해서 사칙 연산을 수행할 필요가 생긴다.
이를 위해 파이토치에서는 자동으로 크기를 맞춰서 연산을 수행하게 만드는 "브로드캐스팅" 기능을 제공한다.
'''
# 일반적인 행렬 덧셈
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)
# 브로드 캐스팅 적용
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# 행렬 곱셈(matmul)과 원소 별 곱셈(mul)의 차이
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2))
print(m1 * m2)
print(m1.mul(m2))

# 평균 구하기
t = torch.FloatTensor([1, 2])
print(t.mean())

# 2차원 행렬 평균 구하기
t = torch.FloatTensor([[1, 2],
                       [3, 4]])
print(t.mean()) # 전체 원소를 대상으로 평균을 구한다.
print(t.mean(dim=0)) # 첫 번째 차원을 제거하고 평균을 구한다. [1, 3]의 평균과 [2, 4]의 평균
print(t.mean(dim=1)) # 두 번째 차원을 제거하고 평균을 구한다. [1, 2]의 평균과 [3, 4]의 평균

# 행렬 덧셈
t = torch.FloatTensor([[1, 2],
                       [3, 4]])
print(t.sum()) # 전체 원소를 대상으로 합을 구한다.
print(t.sum(dim=0)) # 첫 번째 차원을 제거하고 합을 구한다. [1, 3]의 합과 [2, 4]의 합
print(t.sum(dim=1)) # 두 번째 차원을 제거하고 합을 구한다. [1, 2]의 합과 [3, 4]의 합

# 최대(Max)와 아그맥스(ArgMax)구하기
t = torch.FloatTensor([[1, 2],
                       [3, 4]])
print(t.max()) # 전체 원소를 대상으로 max를 구한다.
print(t.max(dim=0)) # 첫 번째 차원을 제거하고 max를 구한다. [1, 3]중 최대 값과 [2, 4]중 최대 값
print(t.max(dim=1)) # 두 번째 차원을 제거하고 max를 구한다. [1, 2]중 최대 값과 [3, 4]중 최대 값
'''
max() 함수는 반환 값이 두 개이다. 값과 idx를 반환해준다.
'''

# 뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경 (중요함)
'''
pytorch의 뷰는 numpy에서의 reshape와 같은 역할을 한다. 즉, 텐서의 크기를 변경해주는 역할을 한다.
'''
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

# ft 텐서를 view를 통하여 2차원 텐서로 변경하기
print(ft.view([-1, 3])) # ft텐서를 (?, 3)의 크기로 변경
'''
view는 기본적으로 변경 전과 후의 텐서 안의 원소의 개수가 유지되어야 한다.
파이토치의 view는 사이즈가 -1로 설정되면, 다른 차원으로부터 해당 값을 유추한다.
'''

# 3차원 텐서의 크기 변경
'''
3차원 텐서에서 3차원 텐소로 차원은 유지하되, 크기(shape)을 바꿔보자.
'''
print(ft.view([-1, 1, 3]))

# 스퀴즈(squeeze) - 1인 차원을 제거
ft = torch.FloatTensor([[0], [1], [2]])
print(ft.size())
print(ft.squeeze())
print(ft.squeeze().size())

# 언스퀴즈(unsqueeze) - 특정 위치에 1인 차원을 추가
ft = torch.FloatTensor([1, 2, 3])
print(ft.size())
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).size())

# 두 텐서를 연결하기 (concatenate)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0)) # dim=0은 차원을 늘리라는 의미를 가진다.
print(torch.cat([x, y], dim=1))

# 스택킹(stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
'''
순차적으로 쌓여 (3 x 2) 텐서가 된다. 
그리고 두 번째 프린트 문과 같이, cat을 이용하여 연결한 것 보다 훨씬 간결해졌다.
'''

# 0과 1로 채워진 텐서
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(torch.ones_like(x)) # x 텐서와 같은 크기이지만 값이 1로만 채워진 텐서를 생성
print(torch.zeros_like(x)) # x 텐서와 같은 크기이지만 값이 0로만 채워진 텐서를 생성

# In-place operation (덮어쓰기 연산)
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))
print(x)
''' 변동 x '''
print(x.mul_(2.))
print(x)