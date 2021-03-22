import torch
import torch.nn as nn # 파이토치에서는 nn.Linear()라는 선형 회귀 모델이 제공된다.
import torch.nn.functional as F # 파이토치에서는 nn.functional.mse_loss()라는 함수로, 평균 제곱 오차를 구해주는 함수를 제공한다.

# 계속 같은 결과가 나오기 위해, 랜덤 시드를 고정한다.
torch.manual_seed(1)

# 단일 선형 회귀 훈련 셋
# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])

# 다중 선형 회귀 훈련 셋
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3, 1)
print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

epochs = 2001
for epoch in range(epochs):
    # H(x) 계산
    prediction = model(x_train)

    # Cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, epochs, cost.item()))
        

# 학습 후 W, b의 결과 보기
print(list(model.parameters())) # 모델에는 가중치 W와 편향 b가 저장되어있다. 이를 출력하기 위함.


# 모델 사용하기
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print(pred_y)

