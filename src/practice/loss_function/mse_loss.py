import torch
import torch.nn.functional as F
import torch.nn as nn


class MseLoss():
    def __init__(self):
        self.x = torch.FloatTensor([[1, 1],
                                    [2, 2]])
        self.x_hat = torch.FloatTensor([[0, 0],
                                        [0, 0]])
    
    def mse_using(self):
        """
        mse 함수를 직접 구현해보자
        """

        def mse(x_hat, x):
            """
            |x_hat| = (batch_size, dim)
            |x| = (batch_size, dim)
            """
            y = ((x - x_hat)**2).mean()
            
            return y

        print(self.x.size(), self.x_hat.size())
        print(mse(self.x_hat, self.x))

    def pytorch_mse(self):
        """
        Pytorch에서 역시나 mse 함수를 제공해준다. 따라서 직접 구할 필요가 없다.
        보통 두 가지로 사용하게 된다.
        
        첫 번째 경우는 import torch.nn.functional의 모듈로,
        어떤 레이어나 뉴런 네트워크 관점이 아닌 위에서 구현한 것처럼 함수로 만들어진 상태. (함수)
        
        reduction인자는 디폴트로 mean으로 되어있다.
        즉, mse_loss함수는 어떤 특정한 스칼라 값을 도출해낼 수 있는 함수인데,
        reduction(차원축소)의 방식을 고를 수 있다.
        mean = 평균
        sum = 합
        none = 차원축소를 안하겠다.
        
        두 번째 경우는 torch.nn의 모듈로, nn모듈을 상속받은 방식이다. (객체)
        """
        # 첫 번째 방식
        print(F.mse_loss(self.x_hat, self.x))
        print(F.mse_loss(self.x_hat, self.x, reduction="sum"))
        print(F.mse_loss(self.x_hat, self.x, reduction="none"))
        
        # 두 번째 방식
        mse_loss = nn.MSELoss()
        print(mse_loss(self.x_hat, self.x))


if __name__ == "__main__":
    test = MseLoss()
    test.pytorch_mse()
