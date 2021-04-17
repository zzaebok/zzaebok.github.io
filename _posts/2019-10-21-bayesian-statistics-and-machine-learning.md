---
title: "베이지안 통계와 머신 러닝"
date: 2019-10-21 21:12:28 -0400
categories: machine_learning
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Intro ##
학부 인공지능 수업, 컴퓨터 비전 수업의 Generative Model들을 배우면서 prior가 어쩌니, 확률분포에서 샘플링을 하니 마니 하는 이야기를 들은 적이 있다.
빈도론자로만 자라온 내게는 와닿지 않는 내용이여서 외우기에 급급했었는데, 오토인코더에 흥미가 생겨 공부하다가 도저히 모르고는 넘어갈 수 없어서 이렇게 정리를 하게되었다.
복잡한 내용이 아니라, 조건부확률과 베이즈 정리 그리고 이러한 내용이 어떻게 머신 러닝과 연결될 수 있는 지에 대해 다뤄보도록 하겠다.
정리하는 내용은 [이 곳](https://towardsdatascience.com/probability-learning-i-bayes-theorem-708a4c02909a) 을 참조하였다.

## 조건부 확률과 베이즈 정리 ##
![bayes theorem](https://www.gigacalculator.com/img/formulas/formula-bayes-theorem.png)

조건부 확률은 말그대로 B가 일어났을 때, A가 일어날 확률을 의미한다. 더 이상 특별할 것이 없다.
이 특별할 것 없어보이는 확률을 특별한 방법으로 구하는 것이 베이즈 정리인데, 도대체 베이즈 정리는 어떤 이유로 사용되는 걸까?
이를 이해하기 위해서는 아주 쉬운 사례를 하나 만들어서 확인해보는 것이 좋다.


>어떤 특정한 암 질병이 있다고 가정하자. 이 암이 걸릴 확률은 0.1%이다. 암이 걸렸는지 안걸렸는지 확인할 수 있는 장치가 하나 있는데, 이 장치는 암이 걸린 사람에게 99% 확률로 양성 반응을 보이고, 암이 걸리지 않은 사람에게 1% 확률로 양성 반응을 보인다고 알려져있다. 만일 내가 이 장치를 이용해 양성 여부를 테스트해봤더니, 세상에 양성이라고 나와버렸다고 생각해보자. 이렇게 '양성 반응을 받았을 때' 내가 실제로 '암에 걸렸을 확률'은 얼마나될까? 대충 어림잡아봐도 90%는 넘을 것 같은 불안감이 생긴다. 과연 그럴까??

위 사례에서 우리는 암에 걸리는 것을 H (hypothesis), 양성 반응이 나온 것을 E (event)라고 부르겠다. 이 때 내가 구하려는 확률은 좌변과 같고
이는 베이즈 정리를 이용하여 우변과 같이 쓸 수 있다.

![bayes_ex1](https://miro.medium.com/max/827/1*4_dSOG3F5qmjOFTGzA829Q.png)

그런데 우변의 분모는 양성 반응이 나올 확률이고 이는 문제를 통해 알 수 없다. 그러나 마찬가지로 조건부확률을 이용하면 이를 다시 쓸 수 있다.

![bayes_ex1_denom](https://miro.medium.com/max/1205/1*uh-9cBH-qsU9z9WS6eK2kw.png)


그러면 우리가 구하고자 하는 확률 $P(H|E)$ 는 문제에서 제시된 $P(H,E), P(H), P(E|H), P(E|~H)$ 들을 이용하여 구할 수 있게 되고 이를 계산하면 다음과 같다.


![bayes_ex1_res](https://miro.medium.com/max/1847/1*wG9EG9D2Vr-gnpbcL-BJ2Q.png)

이를 계산하면 몇 퍼센트의 확률이 나올까? 바로 9%이다. 엄청나게 낮은 확률이다. 처음에 문제를 읽고 한 90%는 넘지 않을까? 생각했던 값과
굉장히 큰 괴리가 있음을 확인할 수 있다. 어쨋든 베이즈 정리는 위의 과정과 같이 이미 알고 있는 확률들을 이용해 원하고자 하는 값을 얻을 수 있다는 큰 장점을 가지고 있다.

하지만 베이즈 통계의 장점은 여기서 끝나지 않는데, 바로 '학습'의 과정이 일어난다는 것이다.
위의 예시를 조금 더 연장시켜보자. 암에 걸렸을 확률이 9%라고 하더라도, 세상에 안심할 수 있는 사람이 몇이나 되겠는가? 걱정이 된 나는
다른 병원에 방문하여 한 번 더 장치를 이용해 암에 걸렸는지 테스트를 해보고자 한다. 그런데 이게 웬일? 또 양성 반응이 나온 것이 아닌가.
그렇다면 이 상황에서 암에 걸렸을 확률은 어떻게 될까??

![bayes_ex2](https://miro.medium.com/max/1847/1*4zQ4vn-ykDurOcOKmOgT1w.png)

다음의 식으로 구할 수 있는데 특이한 점이 있다. 우리는 이전에 암에 걸릴 확률이 0.1%밖에 되지 않는 다는 것을 받아들였지만 지금은 어떠한가?
내가 암에 걸릴 확률은 이제 0.1%가 아니라 이전 테스트의 결과로 인해 9%가 되었다. 따라서 사전확률로 대표되는 P(H)가 업데이트 되었고, 이제는
이 값을 이용해서 내가 암에 걸렸을 확률을 구해야한다. 위 식의 결과는? 안타깝게도 91%가 된다.

위와 같이 베이즈 통계의 핵심은 우리의 데이터와 evidence가 많아지면 많아질수록 우리의 지식이 '업데이트'되고 '향상'되는 데에 있다.
드디어 베이즈 통계를 공부하며 항상 들었던 'posterior를 이용해 prior를 업데이트한다'가 무슨 뜻인지 이해하게 되었다.

![prior and posterior](https://luminousmen.com/media/data-science-bayes-theorem-2.jpg)

맨 처음 prior는 암에 걸릴 확률이라고 알고 있던 0.1%였다. 그러나 양성 반응이라는 이벤트 이후에 사후 확률을 9%로 얻게 되었고, 다시 테스트를 할 때에는
prior가 0.1%가 아닌 9%를 사용하게 되고(양성이 나왔다는 사실을 반영해야 하므로) 사후확률 91%를 얻게 되었다.

## 베이즈 정리와 머신 러닝 ##
그렇다면 이 베이즈 정리와 머신 러닝은 어떠한 문맥에서 공통점을 공유할 수 있을까?
머신 러닝을 대분류로 분류해보면 크게 regression과 classification으로 나타낼 수 있다.
먼저 Regression 문제를 살펴보자. 가장 기본적인 Linear regression을 보면 독립변수와 종속변수의 관계에 대해 추론하는 것이라고 할 수 있다.

![linear regression](https://miro.medium.com/max/1182/1*4Y-w0Em_qLcIdIxKBDDnkQ.png)

이러한 식을 통해 만든 추정치 $$\hat{y}$$와 실제 $$y$$의 차이 (Loss)를 최소화 하는 것이 이 regression의 목표이다.
머신 러닝은 Gradient Descent와 같은 알고리즘을 통해 '점진적으로' 학습하여 parameter를 찾아간다.
그런데 조금 시선을 바꿔서 우리가 추정하고자 하는 $$\theta_0$$과 $$\theta_1$$이 하나의 특정한 값을 갖는 것이 아니라
분포를 갖는다고 생각해보자.
그렇게 하면 우리는 머신 러닝이 parameter를 찾는 과정을 베이즈 정리를 이용해서 표현할 수 있게 되는데 다음과 같다.

![machine learning bayes](https://miro.medium.com/max/480/1*gdgddVSaJQ_BXWJJNYtZ9g.png)

즉, 우리는 P(model)이라는 prior를 알고 있는데 새로운 data가 관측이 되면 posterior(P(model|data))를 얻고 이를
다음번 학습의 prior로 사용하면서 점진적으로 P(model), 즉 parameter들의 분포를 찾아가는 과정이 머신러닝 과정인 것이다.

![MAP](https://miro.medium.com/max/486/1*KmnRZ_zc_cD7CIWylEyrFg.png)

Classification에서도 regression과 다를 바가 하나도 없다.

![machine learning bayes classification](https://miro.medium.com/max/502/1*c63H7VlsTrcntMc5P2v7aw.png)

이렇게 Prior를 업데이트 하는 것이 뭐가 중요하냐라고 생각할 수 있지만,

![male vs female](https://miro.medium.com/max/884/1*okTibKIXXCSLC3ZuKqKwPQ.png)

위 그림과 같이 어떤 classifier가 prior를 고려하는 지 안하는지의 유무에 따라 분류기 성능이 크게 갈릴 수 있다.
단순히 표본을 이용한 MLE의 결과보다 prior를 고려하는 베이지안 방법이 조금 더 맞는 설명일 것이라.
