import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

'''
파이토치의 데이터셋을 만들었다면 데이터 로더를 사용이 가능하다.
데이터로더는 기본적으로 2개의 인자를 입력받는다.
하나는 데이터 셋, 미니 배치의 크기이다. 이때 미니 배치의 크기는 통상적으로 2의 배수를 사용한다. (64, 128, 256 . . . ) 그리고 추가적으로 많이 사용되는 인자로 shuffle이 있다.
shuffle = True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다.
'''
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

epochs = 20
for epoch in range(epochs):
    for batch_idx, samples in enumerate(dataloader):
        '''
        미니 배치와 배치 크기에 대해서 이터레이션을 정의할 수 있다.
        이터레이션은 한 번의 에포크 내에서 이루어지는 매개변수인 가중치 W와 b의 업데이트 횟수이다. 전체 데이터가 2,000일 때 배치 크기를 200으로 한다면,
        이터레이션의 수는 총 10개이다. 이는 한 번의 에포크 당 매개변수 업데이트가 10번 이루어짐을 의미한다.
        '''
        x_train, y_train = samples
        
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(epoch, epochs, batch_idx + 1, len(dataloader), cost.item()))