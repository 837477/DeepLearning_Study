'''
1차 함수로 선형 회귀를 표현하면 y = wx + b의 식으로 그래프로 표현할 수 있다.

머신러닝은 최상의 w(기울기) 와 b(절편)을 구하여 다음 값들을 예측할 수 있다.

즉, w의 값과 b의 값을 찾아내면 다음 x값이 주어지게 된다면 이 값을 예측할 수 있다.
'''


from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

# 당뇨병 데이터 준비하기
diabetes = load_diabetes()

# 입력과 결과 데이터의 크기 확인
print(diabetes.data.shape, diabetes.target.shape)

# 입력 데이터 자세히 보기
print(diabetes.data[:3])

# 출력 데이터 자세히 보기
print(diabetes.target[:3])

# 훈련 데이터 준비하기
# 맷플로립의 scatter()함수로 산점도 그리기
x = diabetes.data[:, 2]
y = diabetes.target
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x_sample = x[99:109]
print(x_sample, x_sample.shape)

'''
결과 값
(442, 10) (442,)
[[ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
  -0.04340085 -0.00259226  0.01990842 -0.01764613]
 [-0.00188202 -0.04464164 -0.05147406 -0.02632783 -0.00844872 -0.01916334
   0.07441156 -0.03949338 -0.06832974 -0.09220405]
 [ 0.08529891  0.05068012  0.04445121 -0.00567061 -0.04559945 -0.03419447
  -0.03235593 -0.00259226  0.00286377 -0.02593034]]
[151.  75. 141.]
[-0.06440781  0.01750591 -0.04500719  0.02828403  0.04121778  0.06492964
 -0.03207344 -0.07626374  0.04984027  0.04552903] (10,)
 '''