'''
단층 퍼셉트론과 다층 퍼셉트론의 차이 이해하기
단층 퍼셉트론은 말 그대로 입력층과 출력층 이 두 개의 레이어로 이루어진 퍼셉트론 구조를 말한다.
단층 퍼셉트론을 이용하여 AND, NAND, OR게이트를 구현할 수 있지만 XOR게이트는 구현할 수 없다.
따라서 중간에 은닉층 레이어를 추가하여 다층 퍼셉트론 구조를 사용해야한다.
'''

# AND 게이트
def AND_GATE(x1, x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1


# NAND 게이트
def NAND_GATE(x1, x2):
    w1 = -0.5
    w2 = -0.5
    b = 0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1


# OR 게이트
def OR_GATE(x1, x2):
    w1 = 0.6
    w2 = 0.6
    b = -0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1

'''
AND, NAND, OR게이트의 단층 퍼셉트론을 그래프로 시각화 해보자.

AND
0 출력 값 : (0, 0), (1, 0), (0, 1)
1 출력 값 : (1, 1)

NAND
0 출력 값 : (1, 1)
1 출력 값 : (0, 0), (1, 0), (0, 1)

OR
0 출력 값 : (0, 0)
1 출력 값 : (1, 1), (1, 0), (0, 1)

위 좌표를 그래프에 점으로 표시하면, 각각 0과 1을 출력을 구분하는 직선 한개를 그을 수 있다.
하지만 XOR은 직선 하나로 이 둘을 구분할 수 없다.

XOR
0 출력 값 : (0, 0), (1, 1)
1 출력 값 : (1, 0), (0, 1)

XOR과 같은 게이트는 직선이 아닌 곡선/비선형 영역으로 분리하면 구분이 가능하다.
따라서 한 개 이상의 은닉층을 사용한 다층 퍼셉트론으로 XOR구현이 가능하다.
(즉, 위에서 사용한 AND, NAND, OR의 퍼셉트론을 조합하여 구현이 가능)
'''

def XOR_GATE(x1, x2):
    hx1 = OR_GATE(x1, x2)
    hx2 = NAND_GATE(x1, x2)
    result = AND_GATE(hx1, hx2)
    return result

if __name__ == "__main__":
    x1 = 0
    x2 = 0
    result = XOR_GATE(x1, x2)
    print(result)
