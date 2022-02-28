import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# - 유방암 데이터 set 준비
###################################################################
# 입력 데이터 확인
print(cancer.data.shape, cancer.target.shape)
print(cancer.data[:3])

# 박스 플롯으로 특성의 사분위 관찰
plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()
print(cancer.feature_names[[3,13,23]])

# 타깃 데이터 확인
print(np.unique(cancer.target, return_counts=True))

# 훈련 데이터 세트 저장
x = cancer.data
y = cancer.target

###################################################################


# - 훈련 set / 테스트 set 나누기
###################################################################
from sklearn.model_selection import train_test_split

# 설정을 매개변수로 지정
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# 결과 확인
print(x_train.shape, x_test.shape)

# 훈련 세트 타깃 확인
print(np.unique(y_train, return_counts=True))

###################################################################


# - 로지스틱 뉴런 클래스 만들기
###################################################################
class LogisticNeuron:

    def __init__(self):
        self.w = None
        self.b = None

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산
        return z

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그래디언트를 계산
        b_grad = 1 * err    # 절편에 대한 그래디언트를 계산
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)     # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])      # 가중치를 초기화
        self.b = 0                        # 절편을 초기화
        for i in range(epochs):           # epochs만큼 반복
            for x_i, y_i in zip(x, y):    # 모든 샘플에 대해 반복
                z = self.forpass(x_i)     # 정방향 계산
                a = self.activation(z)    # 활성화 함수 적용
                err = -(y_i - a)          # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w -= w_grad          # 가중치 업데이트
                self.b -= b_grad          # 절편 업데이트

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]    # 정방향 계산
        a = self.activation(np.array(z))        # 활성화 함수 적용
        return a > 0.5

print(np.zeros((2, 3)))

# 로지스틱 모델 훈련
neuron = LogisticNeuron()
neuron.fit(x_train, y_train)

# 모델 정확도 평가
print(np.mean(neuron.predict(x_test) == y_test))

###################################################################


# - 단일층 신경망 클래스 만들기
###################################################################
class SingleLayer:

    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산
        return z

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그래디언트를 계산
        b_grad = 1 * err    # 절편에 대한 그래디언트를 계산
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)     # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])               # 가중치를 초기화
        self.b = 0                                 # 절편을 초기화
        for i in range(epochs):                    # epochs만큼 반복
            loss = 0
            # 인덱스를 섞습니다
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:                      # 모든 샘플에 대해 반복
                z = self.forpass(x[i])             # 정방향 계산
                a = self.activation(z)             # 활성화 함수 적용
                err = -(y[i] - a)                  # 오차 계산
                w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                self.w -= w_grad                   # 가중치 업데이트
                self.b -= b_grad                   # 절편 업데이트
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            # 에포크마다 평균 손실을 저장
            self.losses.append(loss/len(y))

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]     # 정방향 계산
        return np.array(z) > 0                   # 스텝 함수 적용

    def score(self, x, y):
        return np.mean(self.predict(x) == y)

# 단일층 신경망 훈련
layer = SingleLayer()
layer.fit(x_train, y_train)
print(layer.score(x_test, y_test))

# 손실 함수 누적 값 확인
plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
show(plt)

###################################################################


# - 사이킷런으로 로지스틱 회귀 수행하기
###################################################################
from sklearn.linear_model import SGDClassifier

# 사이킷런으로 훈련 및 평가
sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)
sgd.fit(x_train, y_train)
print(sgd.score(x_test, y_test))

# 사이킷런으로 예측
print(sgd.predict(x_test[0:10]))