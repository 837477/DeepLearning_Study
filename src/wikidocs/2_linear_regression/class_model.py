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


'''
클래스 형태의 모델은 nn.Mudule을 상속받는다. 그리고 __init__()에서 모델의 구조와 동적을 정의하는 생성자를 정의한다.
super()함수를 부르면, 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화 된다.
forward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수이다.
이 forward() 함수는 model 객체를 데이터와 함께 호출한다면 자동으로 실행된다.
예를들어 model이란 이름의 객체를 생성 후, model(입력 데이터)와 같은 형식으로 객체를 호출하면 자동으로 forward 연산이 수행된다.
'''
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()

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

