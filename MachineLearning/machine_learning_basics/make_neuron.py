import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
    
# 훈련 데이터를 만듭니다.
x = diabetes.data[:, 2]
y = diabetes.target

class Neuron():
    def __init__(self):
        self.w = 1.0
        self.b = 1.0
    
    def forpass(self, x):
        '''
        정방향 계산 만들기
        오차를 계산하기 위 해 y_hat = w * x + b 를 구한다.
        '''
        y_hat = x * self.w + self.b
        return y_hat
    
    def backprop(self, x, err):
        '''
        역방향 계산(역전파) 만들기
        eL(손실함수) / ew = -(y - y_hat) * x
        eL / eb = -(y - y_hat)
        '''
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def fit(self, x, y, epochs=100):
        for i in range(epochs):                             # 에포크 반복
            for x_i, y_i in zip(x, y):                      # 모든 샘플에 대해 반복
                y_hat = self.forpass(x_i)                   # 정방향 계산
                err = -(y_i - y_hat)                        # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err)    # 역방향 계산
                self.w -= w_grad                            # 가중치 업데이트
                self.b -= b_grad                            # 절편 업데이트


if __name__ == "__main__":
    neuron = Neuron()
    neuron.fit(x, y)

    plt.scatter(x, y)
    pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
    pt2 = (0.15, 0.15 * neuron.w + neuron.b)
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()