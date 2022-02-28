import torch
import numpy as np


class TensorAllocation():
    def __init__(self):
        print("TensorAllocation")
    
    def float_tensor(self):
        """
        실수형 텐서
        가장 많이 사용하는 텐서, 주로 값을 다룰 때 사용한다.
        """
        tensor = torch.FloatTensor([[1, 2],
                                    [3, 4]])
        print(tensor)

    def long_tensor(self):
        """
        정수형 텐서
        주로 Index 값을 다룰 때 주로 사용한다.
        """
        tensor = torch.LongTensor([[1, 2],
                                   [3, 4]])
        print(tensor)
    
    def byte_tensor(self):
        """
        바이트형 텐서
        주로 True / False를 표현하기 위해 사용한다.
        과거에는 바이트 텐서를 통해 참 / 거짓을 표현했지만,
        현재는 Bool형 텐서도 나왔기 때문에 사용하는 빈도가 줄었다.
        """
        tensor = torch.ByteTensor([[1, 0],
                                   [0, 1]])
        print(tensor)

    def etc_tensor(self):
        """
        값은 랜덤이고, 원하는 사이즈로만 텐서를 생성할 때 사용하는 방법
        Float, Long, Byte 등 어떤 자료형을 사용해도 상관 X      
        """
        tensor = torch.FloatTensor(3, 2)
        print(tensor)
        

class NumPyCompatibility():
    def __init__(self):
        """
        Numpy와 Tensor는 적합성(호환성)이 매우 뛰어남.
        (초기 기본 값은 Numpy 배열로 선언됨)
        """
        self.x = np.array([[1, 2],
                           [3, 4]])
    
    def to_tensor(self):
        """
        Numpy array를 Tensor로 변경
        """
        self.x = torch.from_numpy(self.x)
        print(self.x, type(self.x))
        
    def to_numpy(self):
        """
        Tensor를 Numpy array로 변경
        """
        self.x = self.x.numpy()
        print(self.x, type(self.x))
    

class TensorTypeCasting():
    def __init__(self):
        """
        Tensor는 같은 형식끼리 연산이 가능하다
        만약, LongTensor와 FloatTensor와 연사을 하면 오류가 발생
        따라서, 타입 변환이 필요
        """
        print("TensorTypeCasting")
        self.tensor = torch.FloatTensor([[1, 2],
                                         [3, 4]])
        
    def to_long_tensor(self):
        """
        LongTensor 타입으로 변환
        """
        self.tensor = self.tensor.long()
        print(self.tensor)
    
    def to_float_tensor(self):
        """
        FloatTensor 타입으로 변환
        """
        self.tensor = self.tensor.float()
        print(self.tensor)


class GetShape():
    def __init__(self):
        """
        Tensor의 사이즈를 다룰 수 있다.
        """
        self.tensor = torch.FloatTensor([[[1, 2],
                                          [3, 4]],
                                         [[5, 6],
                                          [7, 8]],
                                         [[9, 10],
                                          [11, 12]]])
    
    def get_size(self):
        """
        Tensor의 크기를 반환 (2가지 방법)
        Numpy와 매우 유사하다.
        """
        print(self.tensor.size())
        print(self.tensor.shape)
    
    def get_dimension(self):
        """
        Tensor의 전체 차원을 반환
        """
        print(self.tensor.dim())
        print(len(self.tensor))
        
    def get_certain_dimension(self):
        """
        Tensor의 특정 차원을 반환
        """
        print(self.tensor.size(1))
        print(self.tensor.shape[1])


if __name__ == "__main__":
    test = GetShape()
    test.get_certain_dimension()
