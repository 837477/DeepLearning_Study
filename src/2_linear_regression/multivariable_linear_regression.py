import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 계속 같은 결과가 나오기 위해, 랜덤 시드를 고정한다.
torch.manual_seed(1)

'''
# 일반적인 방식 ###############################################################
# 훈련 데이터
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b 선언
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
############################################################################
'''

# 행렬 곱셈, 백터의 내적 방식 ####################################################
# 훈련 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b 선언
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)
############################################################################

epochs = 100000
for epoch in range(epochs):
    # 일반적인 방식
    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # 행렬 곱셈, 백터의 내적 방식
    hypothesis = x_train.matmul(W) + b

    cost = torch.mean((hypothesis - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        # Epoch 2000/2001 W: 1.997, b: 0.006 Cost: 0.000005
        # print("Epoch %4d/%d w1: %.3f w2: %.3f w3: %.3f, b: %.3f, Cost: %f" %(epoch, epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()))
        print("Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}".format(epoch, epochs, hypothesis.squeeze().detach(), cost.item()))

'''
Epoch 98000/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166444
Epoch 98100/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166438
Epoch 98200/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166428
Epoch 98300/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166419
Epoch 98400/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166416
Epoch 98500/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166412
Epoch 98600/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166403
Epoch 98700/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166393
Epoch 98800/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166382
Epoch 98900/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166379
Epoch 99000/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166373
Epoch 99100/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166361
Epoch 99200/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166355
Epoch 99300/100000 w1: 1.035 w2: 0.515 w3: 0.462, b: 0.069, Cost: 0.166349
Epoch 99400/100000 w1: 1.035 w2: 0.515 w3: 0.461, b: 0.069, Cost: 0.166341
Epoch 99500/100000 w1: 1.035 w2: 0.516 w3: 0.461, b: 0.069, Cost: 0.166338
Epoch 99600/100000 w1: 1.036 w2: 0.516 w3: 0.461, b: 0.069, Cost: 0.166325
Epoch 99700/100000 w1: 1.036 w2: 0.516 w3: 0.461, b: 0.069, Cost: 0.166323
Epoch 99800/100000 w1: 1.036 w2: 0.516 w3: 0.461, b: 0.069, Cost: 0.166316
Epoch 99900/100000 w1: 1.036 w2: 0.516 w3: 0.461, b: 0.070, Cost: 0.166302

Epoch 98000/100000 hypothesis: tensor([151.5006, 184.6406, 180.6598, 196.1331, 141.9765]) Cost: 0.166444
Epoch 98100/100000 hypothesis: tensor([151.5006, 184.6405, 180.6598, 196.1330, 141.9766]) Cost: 0.166434
Epoch 98200/100000 hypothesis: tensor([151.5007, 184.6405, 180.6599, 196.1328, 141.9767]) Cost: 0.166437
Epoch 98300/100000 hypothesis: tensor([151.5008, 184.6404, 180.6599, 196.1327, 141.9767]) Cost: 0.166423
Epoch 98400/100000 hypothesis: tensor([151.5009, 184.6404, 180.6600, 196.1326, 141.9769]) Cost: 0.166411
Epoch 98500/100000 hypothesis: tensor([151.5010, 184.6404, 180.6600, 196.1324, 141.9769]) Cost: 0.166405
Epoch 98600/100000 hypothesis: tensor([151.5011, 184.6403, 180.6601, 196.1323, 141.9771]) Cost: 0.166400
Epoch 98700/100000 hypothesis: tensor([151.5011, 184.6403, 180.6601, 196.1321, 141.9772]) Cost: 0.166388
Epoch 98800/100000 hypothesis: tensor([151.5012, 184.6402, 180.6602, 196.1320, 141.9772]) Cost: 0.166388
Epoch 98900/100000 hypothesis: tensor([151.5013, 184.6402, 180.6602, 196.1318, 141.9773]) Cost: 0.166378
Epoch 99000/100000 hypothesis: tensor([151.5014, 184.6402, 180.6602, 196.1317, 141.9774]) Cost: 0.166369
Epoch 99100/100000 hypothesis: tensor([151.5015, 184.6401, 180.6603, 196.1316, 141.9775]) Cost: 0.166363
Epoch 99200/100000 hypothesis: tensor([151.5016, 184.6401, 180.6603, 196.1314, 141.9776]) Cost: 0.166353
Epoch 99300/100000 hypothesis: tensor([151.5017, 184.6400, 180.6604, 196.1313, 141.9777]) Cost: 0.166349
Epoch 99400/100000 hypothesis: tensor([151.5017, 184.6400, 180.6604, 196.1312, 141.9778]) Cost: 0.166342
Epoch 99500/100000 hypothesis: tensor([151.5018, 184.6400, 180.6605, 196.1310, 141.9779]) Cost: 0.166337
Epoch 99600/100000 hypothesis: tensor([151.5019, 184.6399, 180.6605, 196.1309, 141.9780]) Cost: 0.166327
Epoch 99700/100000 hypothesis: tensor([151.5020, 184.6399, 180.6606, 196.1308, 141.9781]) Cost: 0.166321
Epoch 99800/100000 hypothesis: tensor([151.5021, 184.6398, 180.6606, 196.1306, 141.9782]) Cost: 0.166313
Epoch 99900/100000 hypothesis: tensor([151.5022, 184.6398, 180.6606, 196.1305, 141.9783]) Cost: 0.166304
'''