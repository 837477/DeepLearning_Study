import torch


class TensorIndexing():
    def __init__(self):
        """
        Tensor의 접근 방법은 일반적으로
        Python list 자료형과 동일하고, 더불어 Numpy array와 같다.
        """
        self.tensor = torch.FloatTensor([[[1, 2],
                                          [3, 4]],
                                         [[5, 6],
                                          [7, 8]],
                                         [[9, 10],
                                          [11, 12]]])
        
    def slicing_access(self):
        """
        기본 값 텐서는 현재 (3, 2, 2) 차원의 텐서이다.
        !! 중요 !! 슬라이싱 한 텐서는 -> 그 특정 부분만 잘려서(복사) 가져온다.
        즉, 차원이 깨진다.
        """
        
        # 첫 번째 dimension(차원)에 접근한다는 뜻이다.
        print(self.tensor[0])
        print(self.tensor[0].size()) # (2, 2)
        print("-"*10)
        
        # 첫 번째 차원만 0번째 dimension을 가져오고 / 나머지(두 번째, 세 번째) 차원은 다 가져와 !
        print(self.tensor[0, :])
        print(self.tensor[0, :].size()) # (2, 2)
        print("-"*10)
        
        # 첫 번째 차원만 0번째 dimension을 가져오고 / 두 번째, 세 번째 차원은 다 가져와 !
        print(self.tensor[0, :, :])
        print(self.tensor[0, :, :].size()) # (2, 2)
        print("-"*10)
        
        # 첫 번째 차원에서 마지막 dimension을 가져오고 다 가져와
        print(self.tensor[-1])
        print(self.tensor[-1].size()) # (2, 2)
        print("-"*10)
        
        # 첫 번째 차원에서 마지막 dimension을 가져오고 / 나머지(두 번째, 세 번째) 차원은 다 가져와 !
        print(self.tensor[-1, :])
        print(self.tensor[-1, :].size()) # (2, 2)
        print("-"*10)
        
        # 첫 번째 차원에서 마지막 dimension을 가져오고 / 두 번째, 세 번째 차원은 다 가져와 !
        print(self.tensor[-1, :, :])
        print(self.tensor[-1, :, :].size()) # (2, 2)
        print("-"*10)
        
        # 첫 번째 차원은 다 가져오고 / 두 번째 차원은 0번째 dimension만 가져오고 / 세 번째 차원은 다 가져와 !
        print(self.tensor[:, 0, :])
        print(self.tensor[:, 0, :].size()) # (3, 2) -> 그림을 그려보면 바로 이해됨.
        print("-"*10)
        
    def range_access(self):
        """
        range 접근은, slicing 접근과 달리 특정 부위만 잘려서(차원이 깨져서) 가져온다.
        즉, 차원이 유지 된 채로 값을 가져온다.
        
        아래의 예시를 보면, 3차원의 텐서 형태는 계속 유지된다.
        """
        print(self.tensor[1:3, :, :].size()) # (2, 2, 2)
        print(self.tensor[:, :1, :].size()) # (3, 1, 2)

    def tensor_split(self):
        """
        Tensor를 특정 size로 자른다.
        
        잘린 Tensor들은 list형태로 반환이된다.
        인자 값으로 dim=x 를 설정할 수 있는데, 이는 특정 dimension(차원)을 기준으로 잡는다. 로 해석할 수 있다.
        
        예를들어서, (10, 4) 차원의 텐서가 존재하고, dim = 0이면,
        첫 번째 차원을 4개씩 분할한다는 뜻이다.
        즉, (4, 4), (4, 4), (2, 4)->나머지 로 반환된다. 
        
        split()과 비슷하게 chunk()라는 함수도 존재하는데,
        이 함수는 개수로 분할 한다.
        = 아직까지는 정확하게 차이점을 잘 모르겠음.
        """

        x = torch.FloatTensor(10, 4)
        splits = x.split(4, dim=0)
        for s in splits:
            print(s.size())
    
    def index_select(self):
        """
        특정 텐서에 대해서 원하는 원소만 속속속 빼올 수 있는 함수이다.
        
        인자 값으로 dim, index를 줄 수 있다.
        index_select(dim=0, index=[2, 1]) 일 경우,
        0번 dimension(첫 번째 차원)에서 인덱스 2와 1인 원소들을 가져와서,
        합.쳐.놓.은 텐서를 반환한다.
        """
        x = torch.FloatTensor([[[1, 1],
                                [2, 2]],
                               [[3, 3],
                                [4, 4]],
                               [[5, 5],
                                [6, 6]]])
        indice = torch.LongTensor([2, 1]) # index 역할을 할 텐서
        print(x.size())
        
        y = x.index_select(dim=0, index=indice)
        print(y)
        print(y.size())

    def tensor_concatenation(self):
        """
        텐서와 텐서를 서로 붙일(합칠) 수 있다.
        
        인자 값으로는 합칠 대상의 텐서 리스트, dim이 있다.
        [x, y]는 합칠 대상의 텐서들을 리스트화 시켜놓은 것이고,
        dim=0은 말 그대로 몇 번째 차원을 대상으로 붙일 것인지 인자로 넘겨주는 것이다.
        
        torch.cat([x, y], dim=0) 같은 경우,
        텐서 x와 y를 첫 번째 차원을 기준으로 서로 붙인다는 뜻이다.
        
        각각의 (3, 3)의 텐서가 존재한다.
        이를 cat()화 시키면,
        (3, 3) + (3, 3)으로 (6, 3)차원의 텐서가 된다.
        
        당연히, 붙일 대상의 차원의 사이즈가 무조건 같아야한다.
        """
        x = torch.FloatTensor([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        y = torch.FloatTensor([[10, 11, 12],
                               [13, 14, 15],
                               [16, 17, 18]])
        print(x.size(), y.size())
        
        result = torch.cat([x, y], dim=0)
        print(result)
        print(result.size())
        
        result = torch.cat([x, y], dim=-1)
        print(result)
        print(result.size())

    def tensor_stack(self):
        """
        텐서와 텐서 사이를 서로 쌓을 수 있다.
        
        concatenation은 같은 차원의 대상으로 서로 붙인다.(합치다.)
        
        stack() 같은 경우는 같은 차원의 대상으로 서로를 붙이는 것이 아니라,
        특정 차원을 대상으로 그 위로 쌓기 때문에 차원이 하나 더 생긴다.
        
        각각의 (3, 3)의 텐서가 존재한다. 이미지화 --> - | -
        이를 stack()화 시키면,
        = <- 이러한 형태로 쌓이게 된다. 즉, (1, 3, 3) + (1, 3, 3)으로 (2, 3, 3)차원이 된다.
        
        마찬가지로 인자 값으로 dim을 설정할 수 있다.
        dim = 0으로 하면 방금 같은 예시와 같고,
        dim - 1으로 하면 (3, 3, 2)차원의 텐서가 생성된다.
        """
        x = torch.FloatTensor([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        y = torch.FloatTensor([[10, 11, 12],
                               [13, 14, 15],
                               [16, 17, 18]])
        print(x.size(), y.size())

        result = torch.stack([x, y])
        print(result)
        print(result.size())
        
        result = torch.stack([x, y], dim=-1)
        print(result)
        print(result.size())

        
        # stack은 사실상 unsqeeze와 cat을 혼용한 함수이다.
        # result = torch.stack([x, y])
        result = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        print(result)
        print(result.size())
        
        # 텐서를 분할 할 때 stack이 유용하게 쓰인다.
        result = []
        for i in range(5): # n이 5가 아니라 10억이라고 치면, 원래는 터진다.
            x = torch.FloatTensor(2, 2) # 따라서, 2개씩 잘라서 통과를 시킨다.
            result += [x] # 그리고 나중에 합쳐야 한다.
        
        result = torch.stack(result) # 이때 유용하게 사용된다.
        print(result)
        print(result.size())
        

if __name__ == "__main__":
    test = TensorIndexing()
    test.tensor_stack()
