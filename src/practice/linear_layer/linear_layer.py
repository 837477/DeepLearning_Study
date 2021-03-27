import torch
import torch.nn as nn


class raw_linear_layer():
    def __init__(self):
        """
        로우 레벨의 linear_layer 방식을 실습한다
        """
        self.x = torch.FloatTensor([[1, 1, 1],
                                    [2, 2, 2],
                                    [3, 3, 3],
                                    [4, 4, 4]])

    def raw(self):
        """
        y = x * W + b 의 식을 로우 레벨로 구현해본다.
        
        원래 가중치(W)와 bias(B)는 랜덤하게 어떤 벨류를 가지고 initialize된다.
        그치만 실습을 위해서 특정 값을 기입해둔 상태이다.
        
        W와 b를 보고 우리는, n = 3, m = 2라는 사실을 알 수 있다.
        따라서,
        |x| = (N, n) // |W| = (n, m) // |xW| = (N, m) = |y|
        """
        W = torch.FloatTensor([[1, 2],
                            [3, 4],
                            [5, 6]])
        b = torch.FloatTensor([2, 2])
        print(self.x.size())
        print(W.size())
        print(b.size())
        
        def linear(x, W, b):
            return torch.matmul(x, W) + b
        
        y = linear(self.x, W, b)
        print(y.size())

    def raw_nn_module(self):
        """
        import torch.nn as nn
        이 모듈에서는 두 가지가 중요하다.
        1. __init__
        2. forward
        이 두 함수가 존재하고, 나중에 자신만의 커스텀 뉴런 네트워크를 만들 때 주로 이 두 함수를 overwriting 하게 된다.
        
        __init__ -> 내가 나중에 forward에서 사용 할, 필요한 것들을 선언해두는 공간
        forward -> 실행 함수, 함수이기 때문에 어떤 x라는 입력이 들어왔을 때 "어떠한 작업"이 실행되어서 y를 배출하듯이,
        이 "어떠한 작업"을 수행하는 공간이다.
        
        +@ foraward는 자동 실행된다.
        """
        
        # 나만의 linear 뉴런 네트워크를 만든다.
        class MyLinear(nn.Module):
            def __init__(self, input_dim=3, output_dim=2):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim

                # 사이즈에 맞게 FLoatTensor을 생성한다.
                self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
                self.b = nn.Parameter(torch.FloatTensor(output_dim))

            # You should override 'forward' method to implement detail.
            # The input arguments and outputs can be designed as you wish.
            def forward(self, x):
                # |x| = (batch_size, input_dim)
                y = torch.matmul(x, self.W) + self.b
                # |y| = (batch_size, input_dim) * (input_dim, output_dim)
                #     = (batch_size, output_dim)
                return y
        
        linear = MyLinear(3, 2)
        y = linear(self.x)
        print(y.size())
        
        for p in linear.parameters():
            print(p)

    def nn_linear(self):
        """
        Pytorch에서는 이미 위에서 구현 한 linear_layer을 이미 제공한다.
        따라서 앞으로 linear 모델을 사용해야 할 때에는 nn.Linear()을 사용하면 된다.
        
        이 모듈은 y = x^T + b의 방식으로 수행된다.
        따라서, 파라미터를 출력해보면, 기존의 로우 레벨로 구현 한 w차원과 다르게,
        (3, 2)차원으로 나오는 것을 확인할 수 있다.
        """
        linear = nn.Linear(3, 2)
        y = linear(self.x)
        print(y.size())
        for p in linear.parameters():
            print(p)

        
if __name__ == "__main__":
    test = raw_linear_layer()
    test.nn_linear()