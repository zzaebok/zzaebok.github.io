---
title: "Entropy와 Mutual Information"
date: 2021-01-24 22:40:28 -0400
categories: Machine_Learning
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## 서론 ##
머신러닝을 공부하다 보면 흔하게 볼 수 있는 개념이 Information theory에 관한 것입니다.
Entropy, Mutual information과 같은 내용들이 나오는데 수식으로만은 이해가 되지 않아 따로 정리를 해보고자 합니다.
직관적이고 간단하게만 정리해보도록 하겠습니다.
이 포스트는 [해당 문서](https://people.cs.umass.edu/~elm/Teaching/Docs/mutInf.pdf)와 [위키피디아](https://en.wikipedia.org/wiki/Mutual_information)를 참고하여 작성합니다.

## Entropy ##
엔트로피는 확률변수의 "unpredictability"를 측정하는 수치라고 생각할 수 있습니다.
unpredictability는 한국어로는 불확실성이라고 할 수 있겠죠.
주사위 A, B가 있다고 가정해봅시다.
A는 우리가 잘 알고 있는 여섯 개의 면이 동일한 확률로 나오는 주사위입니다.
B는 90%확률로 2가 나오고 2%확률로 나머지 5개의 숫자가 나오는 주사위 입니다.
어떤 주사위가 더 불확실한가요?
B이죠. 따라서 주사위의 눈이라는 확률변수에 대해서는 주사위 B의 엔트로피가 더 높게 됩니다.

하지만 엔트로피라는 것은 모호한 개념이 아닙니다.
왜냐하면 정밀한 수학적 정의가 존재하기 때문입니다.
확률 변수 X에 대한 엔트로피는 아래와 같이 쓸 수 있습니다.

$ H(X) = -\sum_{x} {P(x)logP(x)} $

앞면과 뒷면이 나올 확률이 동일한 동전의 예를 들어보겠습니다.

$ P(X=head) = \frac{1}{2} $
$ P(X=tail) = \frac{1}{2} $

이 때의 엔트로피를 구해보면, 1이 나옵니다.

$ H(X) = -[\frac{1}{2}log\frac{1}{2} + \frac{1}{2}log\frac{1}{2}] = 1 $

참고로, $ 0log0 = 0 $ 인 것을 알아두면 좋습니다.


## Joint Entropy ##
Probability에서 Joint probability라는 말이 있듯, Entropy에도 Joint entropy라는 말이 있습니다.
하지만 이는 어려울 것 없이 Joint probability에서 entropy를 구하면 되는 것입니다.
확률변수 X는 맑은 지 (sunny), 비가 오는지 (rainy)에 대한 확률변수, Y는 더운 지 (hot), 추운 지 (cool)에 대한 확률변수라고 합시다.
이 때 joint distribution이 아래와 같이 주어졌다고 가정해보면,

$ P(sunny, hot) = \frac{1}{2} $

$ P(sunny, cool) = \frac{1}{4} $

$ P(rainy, hot) = \frac{1}{4} $

$ P(rainy, cool) = 0 $

주어진 Joint probability를 이용하여 Entropy를 구하기만 하면 Joint entropy를 구한 것이 됩니다.

$ H(X,Y) = -[\frac{1}{2}log\frac{1}{2} + \frac{1}{4}log\frac{1}{4} + \frac{1}{4}log\frac{1}{4} + 0log0] = \frac{3}{2} $



## Mutual Information ##
Mutual information은 두 개의 확률 변수와 관련있는 내용입니다.
특별히, 하나의 확률 변수가 다른 확률변수에 "얼마나 영향을 미치는가"를 측정할 수 있습니다.
또는 "불확실성을 얼마나 줄일 수 있는가"에 대한 개념으로도 이해할 수 있습니다.
왜냐하면 하나의 확률 변수가 다른 확률변수를 결정하는 데에 크게 기여한다면, 하나의 확률변수만 알아도 다른 확률변수의 값을 쉽게 알 수 있기 때문에 그만큼 불확실성이 줄어든다고 해석할 수 있기 때문입니다.

예를 들어 확률 변수 X는 주사위의 눈을 나타내고, Y는 주사위의 눈이 홀수인지 짝수인지를 나타낸다고 해봅시다.
명확히도, Y의 값은 X의 값을 어느정도는 설명할 수 있고, 반대도 마찬가지일 것입니다.
즉, 두 확률 변수는 mutual information을 공유하고 있는 것이지요.

반대로, X, Z가 서로 다른 두 개 주사위의 눈을 나타내는 확률변수일 때는 서로의 mutual information이 존재하지 않습니다.
따라서 우리는 두 개의 변수가 통계적으로 "독립"일 때(P(X,Y) = 0)에 mutual information이 0이 되는 것을 확인할 수 있습니다.

수식으로는 아래와 같습니다.

$ I(X; Y) = \sum_{x}\sum_{y}P(X,Y)log\frac{P(X,Y)}{P(X)P(Y)} $

이 수식의 우변을 계속 바꾸어 나가면 mutual information 역시 Entropy로 표현할 수 있습니다.

$$ = \sum_{x}\sum_{y}P(X,Y)log\frac{P(X,Y)}{P(X)} - \sum_{x}\sum_{y}P(X,Y)logP(Y) $$

$$ = \sum_{x}\sum_{y}P(X)P(Y|X)logP(Y|X) - \sum_{x}\sum_{y}P(X,Y)logP(Y) $$

$$ = \sum_{x}P(X)\sum_{y}P(Y|X)logP(Y|X) - \sum_{y}(\sum_{x}P(X,Y))logP(Y) $$

$$ = \sum_{x}P(X)H(Y|X) - \sum_{y}P(Y)logP(Y) $$

$$ = -H(Y|X) + H(Y) $$

$$ = H(Y) -H(Y|X) $$

이를 직관적으로 해석해보자면 다음과 같습니다.
Entropy를 설명드릴 때 "불확실성"이라는 단어로 설명을 드렸습니다.
따라서 H(Y|X)는 X가 알려졌음에도 남아있는 Y의 불확실성입니다.
즉, mutual information이라는 것은 Y의 불확실성에서 X가 알려졌음에도 남아있는 Y의 불확실성을 뺀 값.
결론적으로 "X를 통해 제거될 수 있는 Y의 불확실성"을 의미하게 되는 것입니다.

물론, mutual information은 그 식에서도 알 수 있듯 Symmetry의 성질을 가지고 있습니다.
따라서 "Y를 통해 제거될 수 있는 X의 불확실성"을 의미할 수도 있으며 처음 설명드렸던 서로에게 "얼마나 영향을 미치는가"라는 의미를 다시 한 번 확인할 수 있습니다.
