import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
'''
파이토치는 데이터셋을 조금 더 쉽게 다룰 수 있도록 torch.utils.data.Dataset과 torch.utils.data.DataLoader를 제공한다.
이를 사용하면 미니 배치 학습, 데이터 셔플, 병렬 처리까지 간단하게 수행할 수 있다.

그런데 torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋을 만드는 경우도 있다.
torch.utils.data.Dataset은 파이토치에서 데이터 셋을 제공하는 추상 클래스이다.
'''

torch.manual_seed(1)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 데이터 셋의 전처리를 해주는 부분
        self.x_data = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
        self.y_data = torch.FloatTensor([[152], [185], [180], [196], [142]])
    
    def __len__(self):
        # 데이터 셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return len(self.x_data)
    
    def __getitem__(self, idx):
        # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

epochs = 20
for epoch in range(epochs):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(epoch, epochs, batch_idx + 1, len(dataloader), cost.item()))