'''
S자 형태로 그래프를 그려주는 시그모이드 함수의 방정식은 다음과 같다.
H(x) = sigmoid(Wx + b) = 1 / (1 + e^-(Wx+b)) = e(Wx + b)

선형 회귀에서는 최적의 W와 b를 찾는 것이 목표였듯이, 로지스틱 회귀 또한 마찬가지이다.
선형 회귀에서는 W가 직의 기울기, b가 y의 절편을 의미했다. 그렇다면 여기에서는 W와 b가 함수의 그래프에 어떤 영향을 주는지 직접 그래프를 그려가며 알아보자
'''

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1/(1+np.exp(-x))


# W가 1이고, b가 0인 그래프
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
'''
위의 그래프를 통해시그모이드 함수는 출력값을 0과 1사이의 값으로 조정하여 반환함을 알 수 있다. 
'''


# W값의 변화에 따른 경사도의 변화
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일때
plt.plot(x, y2, 'g') # W의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
'''
자세히 보면 W 의 값에 따라 그래프의 경사도가 변하는 것을 볼 수 있다. 
앞서 선형 회귀에서 가중치 W는 직선의 기울기를 의미했지만, 여기서는 그래프의 경사도를 결정한다.
W의 값이 커지면 경사가 커지고 W의 값이 작아지면 경사가 작아진다.
'''


# b값의 변화에 따른 좌, 우 이동
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
plt.plot(x, y2, 'g') # x + 1
plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# x_train과 y_train을 텐서로 선언
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)


epochs = 1000
for epoch in range(epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + 
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, cost.item()))