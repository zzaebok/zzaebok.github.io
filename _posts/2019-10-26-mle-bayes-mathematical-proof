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

그림과 같이 각 class에 해당하는 distribution을 추정하고 새로운 $$\X_new$$라는 데이터를 probability density에 따라 여성으로 분류할 수 있겠다.

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




