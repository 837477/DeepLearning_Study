import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 계속 같은 결과가 나오기 위해, 랜덤 시드를 고정한다.
torch.manual_seed(1)


# 변수 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])


# 가중치와 편향의 초기화
'''
선형 회귀는 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 것이다.
그리고 가장 잘 맞는 직선을 정의하는 것은 바로 W와 b로 결정된다.
선형 회귀의 목표는 가장 잘 맞는 직선을 정의하는 W와 b를 찾는 것이다.

현재 가중치는 y = 0x + 0 인 상태로 초기화 했다.
'''
W = torch.zeros(1, requires_grad=True) # 가중치 W를 0으로 초기화 하고, requires_grad는 학습을 통해 값이 변경 되는 변수임을 명시한다.
b = torch.zeros(1, requires_grad=True) # b 또한 W와 같이 초기화 해준다.


# 경사 하강법 구현하기
optimizer = optim.SGD([W, b], lr=0.01) # lr은 학습률(learning rate)를 의미한다.

# 에포크 설정
epochs = 2001
for epoch in range(epochs):
    # 가설 세우기 = H(x)를 계산
    hypothesis = x_train * W + b


    # 비용 함수 선언하기 = cost를 계산
    cost = torch.mean((hypothesis - y_train)**2)


    # 경사 하강법 구현 = cost로 H(x)를 개선
    '''
    optimizer.zero_grad()를 실행하므로서 미분을 통해 얻은 기울기를 0 으로 초기화한다.
    기울기를 초기화 해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있다. (왜냐하면 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기에 계속 누적시키는 특징이 있다.)
    그 다음 경사 하강법 최적화 함수 optimizer.step() 함수를 호출하여 인수로 들어갔던 W와 b에서 리턴되는 변수들의 기울기에 학습률(lr) 0.01을 곱하여 빼줌으로써 업데이트한다.
    '''
    optimizer.zero_grad() # gradient를 0으로 초기화
    cost.backward() # 비용 함수를 미분하여 gradient 계산
    optimizer.step() # W와 b를 업데이트

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(epoch, epochs, W.item(), b.item(), cost.item()))


'''
Epoch    0/2001 W: 0.187, b: 0.080 Cost: 18.666666
Epoch  100/2001 W: 1.746, b: 0.578 Cost: 0.048171
Epoch  200/2001 W: 1.800, b: 0.454 Cost: 0.029767
Epoch  300/2001 W: 1.843, b: 0.357 Cost: 0.018394
Epoch  400/2001 W: 1.876, b: 0.281 Cost: 0.011366
Epoch  500/2001 W: 1.903, b: 0.221 Cost: 0.007024
Epoch  600/2001 W: 1.924, b: 0.174 Cost: 0.004340
Epoch  700/2001 W: 1.940, b: 0.136 Cost: 0.002682
Epoch  800/2001 W: 1.953, b: 0.107 Cost: 0.001657
Epoch  900/2001 W: 1.963, b: 0.084 Cost: 0.001024
Epoch 1000/2001 W: 1.971, b: 0.066 Cost: 0.000633
Epoch 1100/2001 W: 1.977, b: 0.052 Cost: 0.000391
Epoch 1200/2001 W: 1.982, b: 0.041 Cost: 0.000242
Epoch 1300/2001 W: 1.986, b: 0.032 Cost: 0.000149
Epoch 1400/2001 W: 1.989, b: 0.025 Cost: 0.000092
Epoch 1500/2001 W: 1.991, b: 0.020 Cost: 0.000057
Epoch 1600/2001 W: 1.993, b: 0.016 Cost: 0.000035
Epoch 1700/2001 W: 1.995, b: 0.012 Cost: 0.000022
Epoch 1800/2001 W: 1.996, b: 0.010 Cost: 0.000013
Epoch 1900/2001 W: 1.997, b: 0.008 Cost: 0.000008
Epoch 2000/2001 W: 1.997, b: 0.006 Cost: 0.000005
'''