---
title: "EM 알고리즘"
date: 2021-01-26 20:51:28 -0400
categories: Machine_Learning
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## 들어가며 ##
Graph clustering (또는 Community detection 이라고 하더군요)을 공부하다 보니 EM algorithm이라는 말이 여러 논문에 등장하는 것을 확인할 수 있었습니다.
EM 알고리즘은 Expectation Maximization 알고리즘으로 각각 Expectation 단계와 Maximization 단계를 이용하여 클러스터링을 이용한다고 합니다.
여러 글을 찾아봐도 뭔가 아리송하여 직관적으로 이해가 되지 않았습니다.
유튜브를 뒤적거리던 중 정말 [좋은 영상](https://www.youtube.com/watch?v=REypj2sy_5U)을 보게 되었고 영상의 자료를 이용하여 해당 내용에 대해 짧게 정리해보고자 합니다.

## Soft clustering ##
Clustering이라는 단어를 들을 때 가장 먼저 생각나는 알고리즘은 무엇인가요?
제 경우 K-means 알고리즘이 가장 먼저 떠오릅니다.
K-means 알고리즘의 경우 K개의 클러스터를 가정한 뒤 각 클러스터의 평균을 기점으로 하여 클러스터링하는 방식입니다.
따라서 하나의 데이터포인트는 하나의 클러스터에만 포함되는 구조이죠.

하지만, 현실의 문제에서 생각해보면 하나의 데이터는 꼭 하나의 클러스터에만 포함되어야할 이유는 없습니다.
예컨대 직장 동료 A의 경우 직장 동료로 묶일 수도, 퇴근 후에는 친구로 묶일 수도 있는 것처럼 말이지요.
따라서 여러 클러스터들이 서로 겹쳐질(overlap) 수 있는 클러스터링을 Soft clustering이라고 합니다.
A가 직장 동료 그룹에 0.6만큼, 친구 그룹에 0.4만큼 걸쳐 있다고 표현할 수 있겠죠.

이 Soft clustering을 위해서, 하나의 클러스터가 각각의 확률 분포를 가지는 상태를 기반으로 하여 EM 알고리즘이 작동합니다.
즉, 단순히 하나의 mean값을 가지고 clustering을 하는 K-means 클러스터링(hard clustering입니다)과 달리 각각의 클러스터(확률분포)는 확률분포의 모수를 갖습니다.
만일 클러스터가 각각 가우시안 분포를 따른다고 가정한다면 mean, variance 값을 가질 수 있겠죠.
따라서 EM 알고리즘을 이용해 찾고자 하는 것은 각 클러스터의 모수(parameters)입니다.


## Observation ##
우리에게 아래와 같이 클러스터를 알고 있는 데이터들이 주어졌다고 가정해봅시다.

<img src="https://i.imgur.com/LfbBCHB.png" width="500">

이 때 각 클러스터(확률분포)의 모수는 어떻게 추정할 수 있을까요?
간단합니다.
노란색 데이터들의 평균과 분산을 구하고, 파란색 데이터들의 평균과 분산을 구하면 됩니다.
질문이라고 할 것도 없어서 민망하네요 '^'

하지만 우리는 어떤 문제를 풀려고 하고 있나요?

<img src="https://i.imgur.com/9zRV8ot.png" width="500">

여러 데이터가 있지만 각 데이터포인트가 어떤 분포(클러스터)에서 나온 것인지 알지 못하는 상황입니다.
이러한 상황에서 어떤 데이터가 어떤 클러스터에 속하는 지 알아내고자 하는 것이지요.

자 그럼, 조금 시각을 달리하여 어떤 누군가가 와서 여기는 2개의 클러스터가 있고 각 클러스터(확률분포)의 모수를 알려준다면 어떨까요?

<img src="https://i.imgur.com/2AztOGM.png" width="500">

그 모수가 맞는지 틀린지와는 관계 없이 우리는 해당 분포를 기준으로 데이터들을 클러스터링할 수 있습니다.
마치 K-means clustering에서 처음 평균값을 임의로 지정하는 것과 같이 말이죠.

각 확률 분포가 주어졌으므로 우리는 이 확률분포에서 각 데이터의 likelihood를 기준으로 클러스터를 구분지어줄 수 있습니다.
이 likelihood는 분포의 모수를 알고 있고 베이즈 정리를 사용하여 아래와 같이 구할 수 있습니다.

$$ P(b|x_i) = \frac{P(x_i|b)P(b)}{P(x_i|b)P(b) + P(x_i|y)P(y)} $$

$$ P(x_i|b) = \frac{1}{\sqrt{2\pi\sigma_b^2}}exp[-\frac{(x_i-\mu_b)^2}{2\sigma_b^2}] $$

여기서 $ P(b) $는 파란색 분포일 확률, $ P(y) $는 노란색 분포일 확률입니다.
이를 이용하면 누군가 분포를 알려줬을 때 각 데이터가 어떤 분포에 속한지를 정할 수 있죠.

<img src="https://i.imgur.com/wN3uGhL.png" width="500">

결국 닭이 먼저냐 달걀이 먼저냐의 문제가 됩니다.
1. 분포를 알면 각 데이터가 어떤 분포에 속한 데이터인지 알 수 있고 (Expectation step)
2. 데이터가 어떤 분포에 속하는 지 알면 데이터로부터 분포를 (분포의 모수를) 알 수 있습니다. (Maximization step)

EM 알고리즘은 랜덤한 확률 분포를 가정한 뒤(마치 K-means 의 시작처럼), 위 두 가지의 목표를 반복적으로 수행하는 알고리즘입니다.
느낌이 K-means와 매우 흡사한데, K-means clustering과 다른 점은 위에서 언급드린 것과 같이 soft clustering이라는 것입니다.
즉, 우리가 구하고자 하는 것은 임의의 데이터가 파란 분포이냐 노랑 분포이냐의 이분법적 접근이 아닌, $ P(b|x_i), P(y|x_i) $를 구해 각각의 클러스터에 어떤 확률로 들어갈 수 있느냐를 구하고자 하는 것이지요.

## EM 알고리즘 ##
천천히 예시를 통해 EM 알고리즘이 어떻게 동작하는 지 알아봅시다.

먼저, 두 개의 클러스터(확률분포, 여기서 가우시안을 가정합니다)가 있다고 가정합시다.

<img src="https://i.imgur.com/045XT87.png" width="500">

그렇다면 모든 데이터들은 아래와 같이 soft cluster들을 부여받게 됩니다. (Expectation)

<img src="https://i.imgur.com/jgHQooe.png" width="500">

위에서 말씀드린 likelihood 계산을 통해 알게된 값이죠.

$$ b_i = P(b|x_i) $$

$$ y_i = P(y|x_i) = 1 - b_i $$

이렇게 각 데이터 포인트에 soft clustering을 하는 Expectation step이 끝나면, 각 데이터 포인트를 이용하여 새로운 두 개의 확률분포를 구하게 됩니다.
즉 새로운 두 개 확률분포의 평균과 분산 값을 찾아야 하는 것이지요.

이 때 EM 알고리즘이 새로운 파란 분포의 평균과 분산을 구하는 방법은 아래와 같습니다. (Maximization)

$$ \mu_b = \frac{b_1x_1 + b_2x_2 + ... + b_nx_n}{b_1 + b_2 + ... + b_n} $$

$$ \sigma_b^2 = \frac{b_1(x_1-\mu_b)^2 + ... + b_n(x_n-\mu_b)^2}{b_1 + ... + b_n} $$

이렇게 새로운 클러스터(확률분포)의 모수들을 알았다면 이제 다시 데이터포인트들에 클러스터를 부여할 수 있게 됩니다.
또한 $ P(b) $와 같은 prior 또한 $ P(b) = \frac{\sum_i{P(b|x_i)}}{n} $와 같은 방식으로 계속 추정할 수 있습니다.

<img src="https://i.imgur.com/ZFPsP58.png" width="500">

<img src="https://i.imgur.com/sNq4TDO.png" width="500">

이런 식으로 계속 Expectation과 Maximization을 반복하면 K-means 알고리즘처럼 더 이상 변하지 않는 단계가 될 것이고, clustering이 완료되게 됩니다.

## 결론 ##
사실 논문을 보면 EM 알고리즘은 ELBO, KL divergence, variational inference 부분과 섞여 나오곤 합니다.
그래서 저는 오히려 EM 알고리즘이 당최 무엇을 하는 지 알기 힘들었고 단순히 EM 알고리즘의 동작원리에만 집중하여 보고싶어 포스트를 남깁니다.
Clustering의 관점에서 클러스터를 확률분포로 해석한 부분도 재미 있었지만, 역시 K를 hyper parameter로 주어야하는 것이 더 큰 문제가 될 것 같네요.
글을 보시는 분들에게 도움이 되었으면 좋겠고 오탈자 / 잘못된 정리에 대한 지적은 언제나 환영입니다!

