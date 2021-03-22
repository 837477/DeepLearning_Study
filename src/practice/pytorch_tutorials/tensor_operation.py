import torch


class ArithmeticOperations():
    def __init__(self):
        """
        Tensor도 차원벡터(행렬)이기 때문에,
        사칙연산이 가능하다.
        """
        self.a = torch.FloatTensor([[1, 2],
                                    [3, 4]])
        self.b = torch.FloatTensor([[2, 2],
                                    [3, 3]])
    
    def general(self):
        """
        각각의 대응대는 텐서의 원소끼리 연산을 진행한다.
        """
        
        # 덧셈
        print(self.a + self.b)
        
        # 뺄셈
        print(self.a - self.b)
        
        # 곱셈
        print(self.a * self.b)
        
        # 나눗셈
        print(self.a / self.b)
        
        # 동치연산
        print(self.a == self.b)
        
        # 불동치연산
        print(self.a != self.b)

        # 제곱연산
        print(self.a ** self.b)


class InplaceOperations():
    def __init__(self):
        """
        흔히 우리가 연산을 수행하면, 새롭게 메모리를 할당해서 그 곳에 결과 값을 대입한다.
        inplace operations은 새롭게 메모리를 할당하지 않고,
        첫 번째 피연산자의 값을 수정하여, 결과 값을 나타낸다.
        이때의 연산자는 "_" Under_score를 붙인다.
        
        -> 학자들은 새로운 메모리를 할당하지 않고, 기존의 메모리 공간을 그대로 사용하기 때문에
        훨씬 더 효율적이다. 라고 말을 하지만 사실상 파이토치 내부에서 알아서 옵티마이징을 잘 하고 있어서 크게 차이는 없다고 한다.
        -> 즉, 속도의 차이는 크게 없다고 한다. (메모리 상으로는 효율적일 수 있음.)
        """
        self.a = torch.FloatTensor([[1, 2],
                                    [3, 4]])
        self.b = torch.FloatTensor([[2, 2],
                                    [3, 3]])
    
    def not_inplace(self):
        # 기존의 a * b와 똑같음
        # 즉, 새로운 메모리 공간을 할당해서 결과 값을 나타낸다.
        print(self.a.mul(self.b))
    
    def inplace(self):
        # 기존 a의 공간이 결과 값으로 수정된다.
        print(self.a)
        print(self.a.mul_(self.b))
        print(self.a)


class DimensionReducingOperations():
    def __init__(self):
        """
        sum() = 내부 원소들을 다 더한다. (스칼라 값으로 반환됨)
        mean() = 내부 원소들의 평균을 구한다. (스칼라 값으로 변환됨)
        
        위 두 함수에는 "dim"이라는 argument를 사용할 수 있다.
        dim을 쉽게 기억하는 방법은,
        이 연산은 즉, dimension_reducing = 차원이 줄어든다. 이기 때문에 없어질 dimension(차원)이라고 생각하면 편하다.
        
        ex)
        [[1, 2],
         [3, 4]]
        첫 번째 차원(세로) = (1, 3), (2, 4)
        두 번째 차원(가로) = (1, 2), (3, 4)
        따라서, sum(dim=0)일 경우, 첫 번째 차원을 줄이겠다는 뜻으로 [sum(1, 3), sum(2, 4)]의 결과가 나온다.
        
        !!! 여기서 중요한 부분은 텐서는 벡터이기 때문에,
        1 2
        3 4 를 dimension=0을 하면,
        
        4 6 이런 형태가 아닌,
        4
        6 이런 형태로 추상화 할 수 있다.
        """
        self.tensor = torch.FloatTensor([[1, 2],
                                         [3, 4]])
    
    def operation(self):
        # 일반적인 연산
        print(self.a.sum())
        print(self.a.mean())
    
        # dimension 인자를 사용한 연산
        print(self.a.sum(dim=0))
        print(self.a.mean(dim=-1))


class BroadcastOperations():
    def __init__(self):
        """
        원래는 같은 모양(형식, 자료형)의 Tensor들 끼리만 연산을 할 수 있다.
        하지만, 파이토치에서 제공하는 Broadcast라는 기능을 통해서
        서로 다른 모양을 가진 Tensor들 끼리 연산이 가능해진다.
        """
        print("BroadcastOperations")
        self.a = torch.FloatTensor([[1, 2],
                                    [3, 4]])
        self.b = torch.FloatTensor([[2, 2],
                                    [3, 3]])
        self.c = torch.FloatTensor([[[1, 2]]])
        self.d = torch.FloatTensor([3,
                                    5])
        self.e = torch.FloatTensor([[1, 2]])
        self.f = torch.FloatTensor([[3],
                                    [5]])
        self.scalar = 1
        
    def general(self):
        """
        일반적인 연산
        (같은 모양, 크기, 자료형 등)
        """
        print("general")
        print(self.a.size())
        print(self.b.size())
        print(self.a + self.b)
    
    def tesnor_scalar(self):
        """
        Tensor와 scalar의 연산
        """
        print(self.a)
        print(self.a.size())
        result = self.a + self.scalar
        print(result)
        print(result.size())
    
    def tensor_vector(self):
        """
        Tensor와 Vector의 연산
        이 경우는 서로 모양이 다르다.
        이 상황에서 broadcast가 없는 차원을 대상으로 차원을 확장시킨다.
        
        # 예시 1)
         a    d
        1 2   3
        3 4 + 5 = ???
        이 경우에서, d 사실상 [3, 5]의 형태로 (1, 2)로 표현할 수 있다.
        원칙이 서로 같은 형태의 텐서끼리만 연산이 되기 때문에,
        여기서 broadcast가 a의 형태로 d텐서를 수정한다
        즉, [[3, 5], [3, 5]]로 (2, 2)형태를 변환시켜서 연산을 진행한다.
        
        따라서,
        [[1+3, 2+5],
         [3+3, 4+5]] 의 결과가 나온다.
         
        # 예시 2)
        c는 3차원으로 (1, 1, 2)의 size()를 가진다.
        여기서 1차원의 d와 연산을 진행하면
        d = (2, )
        d^ = (1, 1, 2)로 변경한다.
        """
        
        # 예시 1)
        print(self.a.size())
        print(self.d.size())
        result = self.a + self.d
        print(result)
        print(result.size())
        
        # 예시 2)
        print(self.c.size())
        print(self.d.size())
        result = self.c + self.d
        print(result)
        print(result.size())
    
    def tensor_tensor(self):
        """
        Tensor와 Tensor의 연산
        
        현재 텐서들의 형태는 다음과 같다.
        [[1, 2]] -> (1, 2)
        [[3],
         [5]] -> (2, 1)
        
        broadcast를 통해서 서로 부족한 차원을 채워 넣는다. (복사하는 느낌)
        [[1, 2],
         [1, 2]]
        [[3, 3],
         [5, 5]]
        이렇게 차원을 맞추고 연산을 진행한다.
        """
        print(self.e.size())
        print(self.f.size())
        result = self.e + self.f
        print(result)
        print(result.size())

    def failure_case(self):
        """
        broadcast가 안되는 경우
        (1, 2, 2)와 (3, 3)은 broadcast가 차원 확장을 못시킨다.
        당연히 (3, 3)에서 1차원 을 더 확장시켜도 남은 3, 3과 2, 2는 맞지 않기 때문에 연산이 불가능
        그렇다고 데이터를 마음대로 삭제해서 차원 축소를 해서도 안된다.
        """
        x = torch.FloatTensor([[[1, 2],
                                [4, 8]]])
        y = torch.FloatTensor([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        print(x.size())
        print(y.size())
        print(x + y)


if __name__ == "__main__":
    test = BroadcastOperations()
    test.failure_case()