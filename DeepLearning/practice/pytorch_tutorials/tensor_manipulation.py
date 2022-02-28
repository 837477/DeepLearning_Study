import torch


class TensorShaping():
    def __init__(self):
        """
        텐서의 모양을 변경할 수 있다.
        Numpy의 reshape와 매우 유사하다.
        
        주의 할 점은 텐서 개수를 맞춰야한다.
        기본 값의 텐서의 개수는 12개이다. 즉, 원소가 12개
        따라서 변경하고자 하는 차원의 원소의 개수도 같아야한다.
        
        기본 값 = 3차원 텐서
        """
        # 3, 2, 2
        self.tensor = torch.FloatTensor([[[1, 2],
                                          [3, 4]],
                                         [[5, 6],
                                          [7, 8]],
                                         [[9, 10],
                                          [11, 12]]])

    def reshape_to_vector(self):
        """
        3차원 텐서를 1차원 Vector로 변경한다.
        """
        print(self.tensor.size())
        print(self.tensor.reshape(12))
        
        # -1 은 알아서 차원 개수에 맞춰서 바꿔라.
        print(self.tensor.reshape(-1))
    
    def reshape_to_matrix(self):
        """
        3차원 텐서를 2차원 matrix로 변경한다.
        """
        print(self.tensor.size())
        print(self.tensor.reshape(3, 4))
        print(self.tensor.reshape(3, -1))
    
    def reshape_to_tensor(self):
        """
        3차원 텐서를 그대로 n차원 텐서로 변경한다.
        """
        print(self.tensor.size())
        print(self.tensor.reshape(3, 1, 4))
        print(self.tensor.reshape(-1, 3, 2))
        
        print(self.tensor.reshape(3, 2, 2, 1))


class TensorSqueezing():
    def __init__(self):
        """
        squeeze는 "쥐어짜다"라는 뜻으로,
        텐서 또한 쥐어 짜듯이, 필요 없는 dimension이 1인 차원을 삭제한다.
        반대로, 삽입도 가능하다.
        즉, 차원이 1인 차원에 대해서 추가 / 삭제를 할 수 있다.
        """
        self.tensor = torch.FloatTensor([[[1, 2],
                                          [3, 4]]])
    
    def squeeze(self):
        """
        (1, 2, 2)의 텐서를 스퀴징하면, 1이 사라져서 (2, 2)로 변경된다.
        인자 값으로 원하는 차원을 넣어주면 해당 차원이 사라진다.
        
        단, 삭제가 가능 한 차원만 작동하고 삭제가 불가능 한 차원을 인자로 넘겼을 시에는 그대로 반환된다. (작동하지 않음)
        """
        
        print(self.tensor.size())
        print(self.tensor.squeeze())
        print(self.tensor.squeeze().size())
        
        print(self.tensor.size())
        print(self.tensor.squeeze(0).size())
        print(self.tensor.squeeze(1).size()) # 작동 X -> 그대로 반환
    
    def unsqueeze(self):
        """
        squeeze의 반대이다.
        즉, 특정 dimension(차원)에 1을 삽입(insert)하는 것이다.
        """
        
        self.tensor = self.tensor.squeeze()
        print(self.tensor.size())
        
        print(self.tensor.unsqueeze(2))
        print(self.tensor.unsqueeze(-1))
        print(self.tensor.reshape(2, 2, -1))


if __name__ == "__main__":
    test = TensorSqueezing()
    test.unsqueeze()