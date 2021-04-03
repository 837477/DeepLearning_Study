import torch
import torch.nn as nn


class UseGpu():
    def __init__(self):
        """
        gpu를 사용해보자.
        
        !!! 철칙 !!!
        -> 텐서는 무.조.건 같은 디바이스의 텐서끼리만 연산이 가능하다.
        즉, CPU에 활성화 된 텐서 -> 일반 RAM에 활성화 된 텐서와
        GPU에 활성화 된 텐서 -> VRAM에 활성화 된 텐서는 서로 연산이 불가능하다.
        
        더 나아가, GPU가 2대 이상이면, 서로 다른 GPU에 할당 된 텐서끼리도 연산이 물론 불가능하다.
        """
        self.x = torch.FloatTensor(2, 2)
        
    def cuda(self):
        """
        GPU에 텐서를 다뤄보자.
        """
        
        # GPU 텐서 선언
        self.y = torch.cuda.FloatTensor(2, 2)
        print(y)
        
        # CPU의 텐서를 GPU로 이동 (정확하게는 복사)
        self.x.cuda()

        # 특정 GPU로 이동 (정확하게는 복사)
        self.x.cuda(device=torch.device('cuda:0'))
        
        # 어느 디바이스에 있는지 확인
        print(self.x.device)
    
    def to(self):
        """
        to는 위의 cuda 방식과 같지만, 최신 버전(?) 방식이다.
        """
        
        # CPU -> GPU
        self.x.to(device=torch.device('cuda:0'))
        
        # GPU -> CPU
        self.x.to(device=torch.device('cpu'))
        self.x.cpu() # 또 다른 방식
    
    def model_move(self):
        """
        CPU에서 선언된 모델을 GPU로 보내자. (반대도 가능)
        """
        
        # CPU에 선언
        linear = nn.Linear(2, 2)
        for p in linear.parameters():
            print(p)
        
        # GPU로 이동
        linear = linear.cuda()
        for p in linear.parameters():
            print(p)
        
        # 다시 CPU로 이동
        linear = linear.to(device=torch.device('cpu')) # 첫 번째 방식
        for p in linear.parameters():
            print(p)

        linear = linear.cpu() # 두 번째 방식
        for p in linear.parameters():
            print(p)
        
        # model은 어느 디바이스에 있는지 확인할 수 없다.
        # 정확하게는 가능은 하지만, pytorch에서 device 매소드를 지원하지 않는다.
        # linear.device 


if __name__ == "__main__":
    test = UseGpu()
    test.conver_cuda()