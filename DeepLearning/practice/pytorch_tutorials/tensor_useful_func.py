import torch


class TensorFuncions():
    def __init__(self):
        """
        Pytorch는 사용성있는 다양한 함수를 제공한다.
        """
        self.x = torch.FloatTensor([[[1, 2]],
                                    [[3, 4]]])
    
    def tensor_expand(self):
        """
        expand() 함수는 인자로 들어온 크기로 넓혀준다.
        
        즉, 텐서 x는 (2, 1, 2)의 크기의 텐서이다.
        이를 expand(*[2, 3, 2])를 실행하게 된다면,
        내부 값을 그대로 복사하여 (2, 3, 2)형태로 만들어준다.
        
        cat()으로 똑같이 구현할 수 있다.
        """
        result = self.x.expand(*[2, 3, 2])
        print(result)
        print(result.size())
        
        cat_result = torch.cat([self.x, self.x, self.x], dim=1)
        print(cat_result)
        print(cat_result.size())

    def tensor_randperm(self):
        """
        randperm() 함수는 random permitation으로 임의의 어떤 수열, 순열에 대해서 만들어준다.
        예를들어, 10이라고 인자를 주게 된다면, 0~9 숫자를 랜덤으로 섞인 텐서를 만들어준다.
        
        심화) index_select()와 같이 사용하게 된다면, suffling이 가능해진다.
        즉, 저번에 다루어보았던 indice에 randperm()의 랜덤 순열 텐서를 넣어주고 index_select()함수에 건내주면 셔플링이 이루어진다.
        """
        result = torch.randperm(10)
        print(result)
        print(result.size())
        
    def tensor_argmax(self):
        """
        argmax()는 argument_max로, 가장 어떤 값의 최대를 만드는 index를 반환한다.
        
        예를들어, torch.randperm(3**3).reshape(3, 3, -1)의 텐서를 만들었다.
        즉, 랜덤 원소가 27개인 (3, 3, -1)의 텐서가 생성된다.
        그리고 이를 argmax(dim=-1)을 수행하게 된다면,
        -1 dimension이니 제일 마지막 차원 중에서 가장 큰 원소의 index를 반환한다.
        """
        temp = torch.randperm(3**3).reshape(3, 3, -1)
        print(temp)
        print(temp.size())
        
        result = temp.argmax(dim=-1)
        print(result)
        print(result.size())
        
    def tensor_topk(self):
        """
        topk()함수는 argmax()함수와 완전 똑같다.
        하지만, 인덱스만 반환하지 않고 index + value의 형태로 반환해준다.
        
        인자로는 k와 dim을 줄 수있다.
        예를들어, x는 (3, 3, 3)이고 topk(x, k=1, dim=-1)일 경우,
        dim=-1 마지막 차원에서, k=1 top 한 개만 반환해줘.
        즉, (3, 3, 1)의 차원으로 반환이 된다.
        여기서 마지막 1은 k와 같아야한다. 왜냐하면 k는 인자로 받은 값이기 때문에 항상 1이 아닐 수 있다.
        
        우리는 앞으로, 마지막에 최대 값 1개를 뽑아야 할 상황이 많이 생긴다.
        이때, argmax()나 topk() 둘 중 어떠한 것을 사용해도 상관은 없다.
        여기서 기억할 점은 topk()는 차원이 살아있다. 만 기억하면 되겠다.
        """
        values, indices = torch.topk(self.x, k=1, dim=-1)
        
        print(values)
        print(values.size())
        print(indices)
        print(indices.size())
        
        print(values.squeeze(-1))
        print(indices.squeeze(-1))
        
        #########################
        # 차원이 살아있다.
        _, indices = torch.topk(self.x, k=2, dim=-1)
        print(indices.size())
        print(self.x.argmax(dim=-1) == indices[:, :, 0])
        
    def tensor_sort(self):
        """
        sort()는 말 그대로 정렬이다.
        
        topk()로도 정렬을 구현할 수 있다.
        예를들어,
        target_dim = -1 이고,
        topk(x, k=x.size(target_dim), largest=True) 수행하면
        x에서 topk를 뽑을 건데, k가 해당 dimension의 size이다.
        그럼. 현재 x는 (3, 3, 3)이고 이거의 -1 dimension은 마지막 차원이고, 이 차원의 사이즈는 3이다.
        그러면 즉, 큰 순서대로 뽑아와라 라고 해석할 수 있다.
        
        +@ 현재 Pytorch의 버그일 수 있는데,
        CPU에서는 topk()방식이 빠르고,
        GPU에서는 sort()방식이 빠르다고 한다..
        - pytorch 1.6 버전까지는 확인됨.
        """
        
        # topk() 방식
        target_dim = -1
        values, indices = torch.topk(self.x,
                                     k=self.x.size(target_dim),
                                     largest=True)
        print(values)
        print(indices)
        
        # sort()로 topk()구하기
        k = 1
        values, indices = torch.sort(self.x, dim=-1, descending=True)
        values, indices = values[:, :, :k], indices[:, :, :k]
        print(values.squeeze(-1))
        print(indices.squeeze(-1))
        
    def tensor_masked_fill(self):
        """
        masked_fill()함수는 말 그대로 masking이 된 곳에, 채워 넣어라. 라고 볼 수 있다.
        
        예를들어, x = (3, 3)의 텐서가 존재한다.
        그리고 (3, 3)텐서(행렬)에서 mask = x > 4를 수행하면,
        broadcast가 수행되어서 같은 (3, 3)의 텐서에 위 조건에 맞게 True / False의 값들이 채워넣어진다.
        그리고 이를 x.masked_fill(mask, value=-1)을 수행하면,
        말 그대로 x텐서에서 mask가 True인 곳에 -1로 채워라 라고 해석할 수 있다.
        
        즉, mask에 대하여 fill한다.
        후에 자연어 처리에서 자주 사용하게 된다고 한다.
        """
        tensor = torch.FloatTensor([i for i in range(3**2)]).reshape(3, -1)
        
        print(tensor)
        print(tensor.size())
        
        mask = tensor > 4
        print(mask)
        
        result = tensor.masked_fill(mask, value=-1)
        print(result)
    
    def tensor_ones_zeros(self):
        """
        ones()나 zeros() 함수는 말 그대로
        특정 모양(크기)의 텐서의 값을 다 1 혹은 0으로 채워서 만들어주는 함수이다.
        
        ones를 만들 건데, 이미 만들어진 텐서와 같은 타입과, 디바이스로 맞춰야할 때가 생긴다.
        즉, 이미 존재하는 x라는 텐서와 같은 형태, 그리고 GPU는 GPU, CPU는 CPU에(디바이스에 따라 또 타입이 조금 다르다고한다.) 따라서 같은 형태로 만들어주어야 한다.
        즉, 연산을 하기 위해서는 타입과 디바이스가 같아야 한다.
        
        그래서 이미 존재하는 x에 맞는 연산을 해야하는 상황이 생길 경우에는 _like를 사용하면된다.
        ones_like(x)
        이렇게 되면, type과 디바이스도 같게 만들어진다.
        
        +@ x텐서는 지금 GPU에 생성이 되어있는데, 새롭게 만든 ones텐서가 CPU에 생성이 되어있고 이 둘을 연산을 진행하게 된다면 오류가 발생한다고 한다.
        """
        print(torch.ones(2, 3))
        print(torch.zeros(2, 3))
        
        print(torch.ones_like(self.x))
        print(torch.zeros_like(self.x))
        

if __name__ == "__main__":
    test = TensorFuncions()
    test.tensor_ones_zeros()