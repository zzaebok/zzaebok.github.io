---
title: "최대 우도 추정과 베이즈 정리의 간단한 수학적 증명"
date: 2019-10-26 16:17:28 -0400
categories: Machine_Learning
---

## Intro ##
[이 포스트](https://zzaebok.github.io/machine_learning/bayesian-statistics-and-machine-learning/) 를 통해 베이즈 정리와 머신러닝 사이의 연관성에 대해서 살펴봤다.
그러나 자세한 수학적인 증명이 없기 때문에 받아들이기 힘든 부분이 있어 간단한 예제와 함께 수학적인 증명을 통하여 어떻게 parameters를 찾는 것인지 알아보고자 한다.

## 최대 우도 추정 ##
최대 우도 추정(Maximum Likelihood Estimation, MLE)은 모수(parameters)를 찾는 데 우도가 가장 큰 지점을 선택한다는 것이다.
다시말해, MLE의 목표는 주어진 데이터를 가장 잘 설명해줄 수 있는 통계적인 분포를 찾아내는 것이다.
이렇게 분포를 찾게 되면 새로운 데이터가 들어왔을 때 같은 분포를 잘 따르는 지 확인할 수 있고, unlabeled 된 데이터를 분류하기 쉬워진다.
신장이 인풋이고 남자인지 여자인지 분류하는 binary classification 문제를 살펴보자.

![bin_class](https://miro.medium.com/max/801/1*twrMncyWo2RV21D_9QVZgA.png)

그림과 같이 각 class에 해당하는 distribution을 추정하고 새로운 Xnew라는 데이터를 probability density에 따라 여성으로 분류할 수 있겠다.

그러나 여기서 중요한 점은 어떻게 우리가 이러한 확률분포를 계산해낼 수 있냐는 것이다.
먼저 우리가 데이터를 모으게 되면, 우리는 이 소중한 데이터를 가지고 Guess를 하게된다.
즉 우리는 데이터가 어떠한 density function의 모양을 갖는지 추측한다.
Density function의 종류에는 가우시안, 지수분포, 포아송 분포 등이 있다.

우리의 예시에서와 같은 키의 분포는 일반적으로 가우시안분포를 따른다고 알려져 있다.
가우시안 분포의 모수(parameters)는 평균과 분산이고 만약 우리에게 학습 데이터가 d개 있다면, d개의 평균과 d(d+1)/2개의 공분산을 구해야한다.
이러한 평균과 분산 모수들의 분포를 $$\theta$$라고 하자.

최대 우도 추정에서 우리가 하고 싶은 것은 바로 우도를 최대화하는 이 모수 분포를 찾고자하는 것이다.
우도 함수는 다음과 같고,

![likelihood](https://miro.medium.com/max/647/1*cXVi6chnuGa5-ENWuKE21g.png)

만일 data point X들이 서로 독립이라면 joint probability는 각 marginal probability의 곱으로 표현될 수 있으므로 다음과 같다.

![likelihood2](https://miro.medium.com/max/648/1*pnVjaUlvjra6qF-Jt1kBuw.png)

그렇다면 위 우도 함수에서 어떤 $$\theta$$를 골라야할까?
이는 우도 함수를 $$\theta$$에 대해 편미분해서 0이 나오는 값(극대화)이 된다.

![편미분](https://miro.medium.com/max/329/1*ya2KadWsbiu4at9A6q_GVA.png)

우리의 키 / 성별 예측의 문제로 돌아가보자.
먼저 가우시안 분포의 확률 밀도 함수는 다음과 같다.

![가우시안](https://miro.medium.com/max/458/1*LWmCtK8YmZXxEbcJthNd0g.png)

일단 데이터 분포의 확률 밀도 함수를 알게 되면 우리는 다음과 같은 방식으로 우도를 계산할 수 있게 된다.

![우도 계산](https://miro.medium.com/max/479/1*9dcCRRNey95whKrFlzqzGg.png)

이는 로그를 취하여도 우도를 최대화하는 모수가 변하지 않으므로,

![모수안변](https://miro.medium.com/max/536/1*JXCF2ZHRppfC4yqbI8eypg.png)

이에 대해 우리가 찾고자 하는 우도 극대화 모수 (여기서는 평균, 분산)을 찾아보면 다음과 같다.

![평균](https://miro.medium.com/max/591/1*vmrP4Uqw6gke_4pNYH5xVw.png)

![분산](https://miro.medium.com/max/605/1*CIDOM88j0DlhEucHa5BvYQ.png)

흥미롭지만 어떻게보면 당연한 결과이다.
평균이 표본평균이고, 분산이 표본분산인 density function에서 sampling을 했을 때 당연히 우리가 가진 데이터의 모양대로 뽑힐 확률이 높았을 것이기 때문이다.

표본으로 추출한 남학생의 키가 다음과 같다고 해보자(n=10)

![남학생](https://miro.medium.com/max/509/1*ZcZUDFReo3vp7CmN12Bc2w.png)

만일 우리가 모수 $$\mu$$를 180, $$\sigma$$를 4라는 임의의 수로 잡는다면 우도의 구성은 다음과 같지만,

![우도1](https://miro.medium.com/max/536/1*W5dk3mAxsEiBdcfmi_CN1A.png)

표본 평균, 표본 분산으로 잡는다면 다음과 같아진다.

![우도2](https://miro.medium.com/max/634/1*B9PDeFyhc-yd5dYJKuGx0A.png)

## 베이즈 정리 ##
이전 포스트에서도 살펴본 대로 베이즈 정리는 모수 $$\theta$$와 데이터 $$X$$를 이용하여 표현할 수 있다.
그리고 베이즈 정리의 강점은 바로 prior (사전 지식)을 우리의 도구로서 활용할 수 있다는 데에 있다.

![베이즈](https://miro.medium.com/max/258/1*tvaribyQUbPz5FBmGbM9FQ.png)

MLE에서 모수를 찾는 방법과 같이 우리는 베이즈 정리에서도 이를 활용할 수 있다.

![베이즈1](https://miro.medium.com/max/444/1*8495SXWOV0IbxOa-xeatVg.png)

식을 살펴보면 최대 우도 추정과 베이즈 정리가 어떤식으로든 조금 연관이 있을 거라는 걸 확인할 수 있다.
극대를 만들어주는 $$\theta$$를 찾기 위해 편미분 해서 0으로 놓아보자.

![편미분2](https://miro.medium.com/max/963/1*-QCsoCYiNflJet0BpE_YNA.png)

두 번째항이 바로 previous knowledge of the model을 뜻하게 되고 바로 베이즈 정리에서 의미를 갖는 부분이다.
이 의미를 제대로 확인하기 위해서 예전에 살펴본 선형회귀식을 상기시켜본다.

![LR](https://miro.medium.com/max/758/1*gxYFlkpODC_P6zxneH1wEw.png)

한 가지 주의할 점은 위의 선형회귀식은 실제 y값을 정확하게 예측하지 못하고 특정한 error가 동반된다는 것이다.

![error](https://miro.medium.com/max/935/1*mqGqbmlnUypn99gggJvC8w.png)

이 에러는 평균이 0, 분산이 $$\sigma^2$$라는 특징을 갖는다. 이 가정이 비현실적인 것 처럼 보이지만 실제로 그런 경우가 훨씬 많다고한다.
잘 찾은 회귀식은 뭘까? 바로 이 error term을 최소화 하는 것이다. 즉 실제 $$y$$ 값과 $$\hat{y}$$의 차이를 최소한다는 것인데,

![MSE](https://miro.medium.com/max/765/1*w7hA0VxfqmSZRvYyLjbmYw.png)

위 식을 최소화한다고 생각해볼 수 있다. 머신러닝 / 딥러닝을 공부해본 사람이라면 알겠지만 이 식은 흔히들
MSE라고 부르는 Loss function의 모양을 하고 있다.

지금부터 위 Loss 식을 최소화하는 것과 MLE를 통해 우리의 데이터에 가장 잘 맞는 모수의 확률 분포를 찾는 것은 동치관계라는 것을 증명해보도록 하겠다.

회귀식에서의 우도함수는 다음과 같다.

![회귀우도](https://miro.medium.com/max/720/1*L6cgyNmBm_g-JDxFo3yX2A.png)

이를 마찬가지로 로그화 해주면

![로그회귀우도](https://miro.medium.com/max/933/1*rPqwO-kbMGHlO5svb1j3OQ.png)

먼저 여기서 우변의 두 번째 항을 살펴보면, $$X$$ 와 $$\theta$$가 static이라고 가정하게 되는데
이 경우 아까 우리가 세웠던 식에 따라 $$y_i$$는 error term의 분포 모양을 그대로 따르게 된다.
즉, $$y_i$$는 평균이 $$f(x_i|\theta)$$이고 분산이 $$\sigma^2$$인 가우시안 분포를 따르게 되는 것이다.

다시말해 data (X)와 parameter를 given으로 하였을 때 $$y_i$$의 조건부 확률은 이러한 꼴이 된다.

![y prob](https://miro.medium.com/max/1153/1*C1HELFGU5_gK-UmQf8oOeg.png)

여기서 constant는 극대화 모수 설정에 영향을 주지 않으므로 1이라고 가정하면 베이즈 정리 중 우도함수에 대한 표현을
아래와 같이 할 수 있게 된다.

![LR lf](https://miro.medium.com/max/1095/1*pbZ9yW-AJK30I0qyg7j-xw.png)

우리는 무엇을 하려고 했는가?
바로 우도 극대화이다. 그렇다면 우도 극대화를 어떻게 달성할 수 있는가?
오른쪽에 보이는 sum of squares를 최소화함으로써 달성할 수 있다. (어차피 $$\theta$$에 대해 미분시 ln(p(x))는 사라지므로)
이렇게 최대 우도 추정과 MSE Loss 극소화가 같은 것임을 확인할 수 있다.

그럼 다시 처음으로 돌아가보자.
우리가 베이즈 정리로 썼던 다음의 식이

![베이즈로가](https://miro.medium.com/max/444/1*8495SXWOV0IbxOa-xeatVg.png)

이렇게 정리가 되게 된다.

![정리](https://miro.medium.com/max/777/1*0Ig0FNtDSUlX8DcwbElqMA.png)

다시말해, 우리가 관찰된 데이터를 가장 높은 확률로 찾게 만들어주는 모델의 모수를 찾기 위해 다음의 식을 극대화하면
최대 우도 추정뿐만 아니라 extra term $$ln p(x)$$도 고려를 해주어야 한다는 것이다. 그리고 바로 이 term이 prior knoledge of the model parameters가 된다.

이 식에서 유의해서 봐야하는 부분은 $$\sigma$$이다. 이것이 무얼 의미했던가? 바로 prediction의 표준편차이다.
prediction할 때 분산이 커지는, 즉 불안정한 때는 언제인가? 학습 초반 데이터가 부족해 제대로 모수를 추정하지
못했을 때 커진다. 그렇다면 위 식이 의미하는 바는 무엇인가? $$\sigma$$가 커지면 첫 번째 항은 매우 작아져 이 prior knowledge의 영향이 커진다. 즉, 학습이 불안정한 때에는 사전 지식이 중요한 역할을 하게 된다는 것이다.

실제 예시를 통해 살펴보자. 
다음과 같은 선형 회귀식에서

![선형회귀3](https://miro.medium.com/max/472/1*2M_wwTiGvCG3BNAjHY_I3A.png)

a는 평균이 0 표준편차가 0.1, b는 평균이 1 표준편차가 0.5라고 할 때 a와 b의 밀도함수는 다음과 같다.

![df](https://miro.medium.com/max/515/1*ihLKxdAhYVGZ_ltC-aOFRg.png)

constant는 제외하고 베이즈 식을 써보면

![베이즈식](https://miro.medium.com/max/905/1*kvC1fPtp4KWVOtEARNVjLA.png)

여기서 a에 대해 위 식을 편미분 해보면

![편미분식](https://miro.medium.com/max/673/1*EByygoPtG0S1y_lwtO8rEg.png)

자 이제 어디가 prior에 의해 영향을 받는 부분인가? 바로 $$100\sigma^2$$이다.
학습이 불안정한 때, 학습 초기의 큰 분산은 파라미터를 작은 값으로 만들어준다.
만일 분산이 작다면 파라미터가 의미있는 큰 값이어도 상관 없지만 분산이 크다면 파라미터를
줄여 모델에 영향을 적게 주는 것이 확실히 좋을 것이다. 이렇게 베이즈는 prior의 반영을 통해
우리가 모델을 학습하는 데에 영향을 끼쳤다고 볼 수 있을 것이다.





