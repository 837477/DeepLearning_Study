import torch


class MatrixMultiplication():
    def __init__(self):
        self.x = torch.FloatTensor([[1, 2],
                                    [3, 4],
                                    [5, 6]])
        self.y = torch.FloatTensor([[1, 2],
                                    [1, 2]])

        self.bx = torch.FloatTensor([[[1, 2],
                                      [3, 4],
                                      [5, 6]],
                                     [[7, 8],
                                      [9, 10],
                                      [11, 12]],
                                     [[13, 14],
                                      [15, 16],
                                      [17, 18]]])
        self.by = torch.FloatTensor([[[1, 2, 2],
                                      [1, 2, 2]],
                                     [[1, 3, 3],
                                      [1, 3, 3]],
                                     [[1, 4, 4],
                                      [1, 4, 4]]])
        
    def matmul(self):
        """
        행렬 곱
        
        행렬 곱은 matmul() 매소드를 이용한다.
        * <- 연산은 텐서와 텐서끼리의 element 곱 연산이기 때문에,
        두 피연산자의 크기(형태, 차원)이 같아야한다.
        """
        
        print(self.x.size(), self.y.size())
        
        result = torch.matmul(self.x, self.y)
        print(result.size())
        print(result)
    
    def bmm(self):
        """
        batch matrix multiplication
        
        bx = (3, 3, 2)의 차원이고, by = (3, 2, 3)이다.
        이 둘의 matrix의 연산은 가능하다.
        index [1], [2]를 보면, (3, 2)와 (2, 3)이기 때문에 행렬 곱 연산이 가능하다.
        그리고 맨 첫 번째 차원은 말 그대로 (3, 2), (2, 3)의 행렬이 각각 3차원으로 3층 쌓여있다. 라고 해석이 가능하며,
        그리고 맨 앞 차원이 동일해야 Parallel하게 연산이 가능하다.
        """
        
        print(self.bx.size(), self.by.size())
        
        result = torch.bmm(self.bx, self.by)
        print(result.size())
        print(result)


if __name__ == "__main__":
    test = MatrixMultiplication()
    test.bmm()