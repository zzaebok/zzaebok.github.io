---
title: "Federated Learning 정리"
date: 2020-12-06 10:18:28 -0400
categories: machine_learning
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## 서론 ##

Federated Learning이란, 한국말로 굳이 번역하자면 '연합 학습'입니다.
오늘은 이 Federated Learning이 어떠한 개념인지, 어떻게 동작하는지, 그리고 또 분산 학습(distributed learning)과는 어떻게 다른지 살펴보겠습니다.
본 글은 [해당 논문](https://arxiv.org/pdf/1602.05629.pdf) 을 참고하여 작성하였습니다.
잘못 작성한 부분이 있다면 언제든지 연락주시면 감사드리겠습니다.

## 배경 ##

최근 들어 딥러닝 알고리즘에 대한 관심이 급증하고 있으며 이를 통해 모바일 단말 상에서 재미있는 서비스들이 많이 생겨나고 있습니다.
예를 들면 사용자의 사용 패턴을 파악하여 키보드 앱에서 다음 입력할 단어들을 추천하는 서비스가 있겠네요.
하지만 모바일 환경에서 딥러닝을 적용하는 것은 쉬운 일이 아닙니다.
기존의 딥러닝 방식을 모바일 환경에서 적용하기 위해서는 데이터를 서버로 모으고, 이 데이터들을 이용해 딥너링 모델을 학습해야합니다.
예시로 드린 키보드앱과 같은 경우 모델을 학습하려면 사용자의 타이핑 기록들을 모두 서버로 올려야하는 데, 이는 현실적으로 불가능하다고 볼 수 있습니다.
누군가의 모든 타이핑 기록을 서버로 보낸다? 어떤 사용자가 이러한 서비스에 동의를 할까요.


그렇다면 현재 배포되어 있는 딥러닝 모바일 앱들은 내 사적인 정보를 서버로 보내는 걸까요?
물론 그럴 수도 있습니다.
하지만 더 현실적인 방법은 프록시 데이터로 학습한 딥러닝 모델을 모바일 환경에 배포하는 것입니다.
이미지를 분류하는 모델이라고 한다면, 제작자가 가지고 있는 데이터(예컨대 이미지넷)를 통해 학습시킨 모델을 TF lite, Pytorch mobile과 같은 라이브를 이용하여 경량화시키고 배포하는 것이지요.
당연하게도 이와 같은 방법들을 이용한다면 현실데이터와는 차이가 크기 때문에 성능이 안좋거나, 모델이 업데이트될 때마다 앱 업데이트를 시켜줘야하는 불편한 상황이 생기기도 합니다.

## Federated Learning이란 ##

따라서 Federated learning은 위와 같은 문제들을 해결하고자 탄생하였다고 할 수 있습니다.
바로 Data를 서버로 보내는 것이 아니라 모바일 환경에 그대로 두는 방법입니다. (Leave the training data on the mobile devices)
만일 Data를 서버로 보내지 않는다면, Privacy 문제가 당연히 해결될 뿐만 아니라 실제 사용자의 데이터를 이용하여 모델을 학습시킬 수 있다는 장점이 있습니다.
물론 깊이 들어가면 Privacy가 완전히 해결되는 것은 아니지만, 기존에 데이터를 서버로 올려보내는 것보다는 훨씬 안전하게 학습이 가능하겠지요.

좀 더 자세하게 말하자면, Data는 모바일 환경에 두고 모바일 컴퓨팅 파워를 이용해 글로벌 모델을 학습하고, 이 '모델 학습의 결과'를 서버에서 종합하는 형태입니다.
즉, 데이터 대신 모델 weights를 서버와 교환하는 형태가 되는 것입니다.

<img src="https://miro.medium.com/max/2560/1*Bbgj7VNT-01KD3U2lWXDgw.png" width="500">

서버에서는 Aggregation model이 동작하며 model weight를 각 모바일 환경에 전달합니다.
각 모바일 디바이스는 해당 환경에서의 데이터를 이용해 weight를 업데이트하고 서버로 전송합니다.
Aggregation model은 다시 이 weights들을 받아 합치고 다시 학습 사이클이 시작되는 것이지요.

하지만 이러한 Federated learning 환경은, 그 Data의 특성이 기존의 딥러닝 환경과는 확연한 차이가 있습니다.
먼저, 데이터들이 Non-IID ([IID](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)) 분포로부터 추출됩니다.
쉽게 말해 누군가의 휴대폰에는 음식 사진만 잔뜩 있는데, 다른 누군가의 휴대폰에는 인물 사진만 잔뜩 있을 수 있겠죠.
그런데 각각 음식사진으로만 학습한 모델, 인물 사진으로만 학습한 모델을 합친다고 좋은 모델이 나올지 확신할 수 없습니다.
또한, 데이터는 굉장히 불균등하게 퍼져있습니다.
누군가는 갤러리에 100장의 사진도 없는 반면, 다른 누군가는 10GB가 넘는 양을 휴대폰에 저장하고 있을 수도 있는 것이지요.

뿐만 아니라, Federated learning에서는 communication의 cost가 가장 중요한 factor입니다.
업데이트된 모델을 네트워크를 통해 교환하는 것이 굉장히 큰 비용인 것이지요.
따라서, 모바일 컴퓨팅 파워에 의한 속도보다는 이 network communication을 줄이는 것이 더 중요하게 됩니다.

## 분산 학습과는 무엇이 다른가요? ##

그렇다면 이 Federated learning과  분산 학습 (distributed learning)은 어떤 차이를 가지고 있을까요?
기본적으로 분산 학습은 데이터를 먼저 중앙에서 모은 다음, 여러 컴퓨팅 리소스로 분산을 시키는 형태입니다.
따라서 위에서 언급했던 Non-IID, 불균형 데이터의 특성을 갖지 않습니다.
왜냐하면 중앙으로 모은 데이터를 균등하게, 그리고 IID한 특성으로 갖도록 분산시킨 뒤에 학습이 진행되기 때문입니다.

## 원리 ##

먼저 기존 딥러닝의 경우 Loss는 아래의 그림과 같은 개념으로 구해지게 됩니다.

![Default_loss](https://i.imgur.com/6O4ksjY.png)

매 trainig item에 대하여 loss를 구하여 평균을 내고 이를 이용해 backpropagation을 진행하는 익숙한 형태입니다.
하지만 이런 식으로 진행되는 학습은 모든 데이터를 중앙에서 가지고 있을 때에나 가능한 것이고, 만일 데이터를 모바일단에 그대로 보관한다면 Loss를 아래와 같이 다시 쓸 수 있습니다.

![FL_loss](https://i.imgur.com/1EQVnSo.png)

여기서 $K$는 client의 수, $F_k$는 각 클라이언트별 Loss를 의미하며, $\mathcal{P}_k$는 클라이언트 k가 가진 데이터들을 의미한다고 볼 수 있습니다.
만일 IID 상황이라면 $F_k(w)$는 $f(w)$의 approximation이 될 수 있지만, 우리의 상황은 Non-IID 상황으로, $F_k$가 굉장히 bad approximation일 수도 있는 상황인 것입니다.

이 때 논문의 저자들이 제시한 FederatedAveraging 알고리즘은 굉장히 심플한데, 아래와 같습니다.

![Algorithm](https://i.imgur.com/H9LQoAQ.png)

먼저 약자들에 대해 설명을 드리겠습니다.
B는 local minibatch size를 의미하고, E는 local epoch의 수를 의미합니다.
만약 B가 무한대라고 표기되어있으면, 모바일 단말의 모든 local data를 사용한다는 의미가 되겠지요.
C는 한 번의 round에 참여시킬 클라이언트의 비율을 의미합니다.
예를 들어 C = 0.1 이고 클라이언트 K = 1000이라고 할 때 한 번의 communication round에 0.1 * 1000 = 100 명의 클라이언트를 참여시키는 것입니다.

자 이제 알고리즘을 다시 살펴보면 굉장히 간단한 것을 확인할 수 있습니다.
communication rount t번 동안 학습을 진행할 것이며 매 round별 일정 클라이언트가 참여해 모델을 업데이트하게 됩니다.
각 클라이언트는 로컬 데이터를 Batch size에 맞게 자른 뒤 로컬 epoch E번 학습을 진행하고 업데이트된 모델 weights를 서버로 보냅니다.
서버는 이러한 weights를 받아 데이터의 개수에 맞게 가중합을 함으로써 모델을 업데이트하게 되는 구조입니다.
매 라운드마다 학습된 결과를 다시 모델들에게 뿌려줌으로서 새로운 라운드가 시작되게 됩니다.

그러나 여기서 궁금증이 드시는 분들도 있으실 것 같습니다.
그저 각자 weight를 initialize시키고 학습한 뒤에 parameter space에서 합치는 것이 잘 working할까요?
정답은 No. 입니다.
MNIST 데이터를 두 개로 분리한 뒤 각자 parameter를 초기화하여 학습한 뒤 합치는 경우에는 그저 하나의 데이터덩어리만 써서 학습한 것보다 loss가 높게 나온답니다.

![Agg_param_space](https://i.imgur.com/iUERhrA.png)

그렇지만 놀랍게도 두 데이터 덩어리의 모델의 초기화를 동일하게 해준다면, 이후에 parameter space에서 합치는 것이 오른쪽 그림처럼 잘 working한다고 하네요.
왼쪽 그림처럼 각자 따로 초기화를 할 경우 하나의 모델보다 합치는 것이 훨씬 안좋은 성능을 보여준다고 합니다.


## 적용 결과 ##

위 논문에서는 해당 FederatedAveraging 알고리즘을 MNIST task, Character prediction task에 적용하였습니다.
Non-IID 분포와 불균형 데이터를 재현하기 위하여, MNIST 데이터는 각 모바일 환경이 최대 2가지의 숫자만 가질 수 있도록 하였고, Character prediction 모델은 셰익스피어의 작품들을 대사 단위로 잘라내어 불균형한 데이터분포를 갖도록 하였습니다.
또한 C, B, E와 같은 변수들을 가지고 다양한 실험을 진행하였습니다.

먼저 Parallelism의 효과입니다.
기본적인 병렬연산은 C를 증가시킴으로써 달성이 가능합니다.
한 번의 라운드에 참여하는 클라이언트들이 증가한다는 것은 병렬 효과를 증가시킨다는 것을 의미하기 때문입니다.
결과를 확인하면 다음과 같습니다.

![increase_C](https://i.imgur.com/kzNw8Fg.png)

MNIST 데이터셋에 대한 결과인데요, 각 칸의 숫자들은 목표한 성능까지 도달하기 위한 round 수를 의미합니다.
생각보다 의외의 결과는 C를 증가시키는 것(병렬을 증가시키는 것)이 빠른 성능 도달에 큰 영향을 주지 못한다는 것입니다.
C가 0일 때는 하나의 클라이언트만 사용한 것인데, 이것이 C=0.1이 될 때에는 성능 개선이 있지만 이 이상의 병렬성 증가가 성능에 큰 영향을 미치지 못하는 것을 확인할 수 있습니다.

두 번째로 확인한 것은 클라이언트 단에서의 computation 증가의 효과입니다.
말씀드렸던 것과 같이, Federated learning에서 제 1로 신경쓰는 것은 communication cost를 낮추는 것입니다.
매 epoch마다 model weights를 업데이트하는 일반적인 딥러닝 방식은 communication cost를 낮추기에 적합하지 않습니다.
local computation을 증가시켜(B를 감소, E를 증가) 많은 양을 local에서 계산한 뒤에 network 통신은 적게 하는 게 좋은 것이지요.

![increase_local_computation](https://i.imgur.com/a86ucNc.png)

그런데, 이렇게 local상에서 계산을 많이하면 많이할 수록 overfitting이 되는 것이 아닌가 생각할 수 있습니다.
하지만 최근의 연구결과에서는 현실상황에서는 over-parameterized Neural Nets이 이전에 생각했던 것 같은 bad local optima에 빠지지 않는다고 합니다.
따라서 적당한 overfitting이 된다면 이러한 잘 학습된? 모델들을 합쳤을 때에도 그대로 잘 작동하는 것을 기대할 수 있습니다.

그렇다면, 극단적으로 local을 학습시킨 뒤에 합치는 것은 어떨까요?
예컨대, E를 굉장히 크게 늘려 로컬 상황에서 극한으로 학습을 시키는 것입니다.

![E_diverge](https://i.imgur.com/fzdsEks.png)

하지만 불행히도 이러한 극단적인 local training은 loss 를 발산시켜버리기까지 한다는 것을 확인할 수 있습니다.
따라서 본 논문에서는 우리가 일반적인 딥러닝에서 learning rate을 점점 감소시키듯이 Federated learning에서는 점차적으로 E를 감소시키는 것이 좋다고 말하고 있습니다.


## Demo ##

Federated learning은 상대적으로 최근에 각광받기 시작한 개념으로 이렇다 쉽게 확인할 수 있는 데모를 찾기 어렵습니다.
하지만 [Openmined](https://www.openmined.org/)라는 Open source가 이러한 환경들을 제작하고 있고 간단한 demo를 돌려볼 수 있도록 제공하고 있습니다.
기본적으로 2~3가지 라이브러리를 이용하여야 하며, 모바일 환경을 이용한 Federated learning 예제는 [이곳](https://github.com/OpenMined/KotlinSyft#quick-start)을 똑같이 따라하시면 됩니다.

[PyGrid](https://github.com/OpenMined/PyGrid)라는 peer to peer network를 이용하여 Data owner, Data scientist가 Federated learning을 할 수 있도록 환경을 제공할 수 있습니다.
이와 함께 모바일 단말, 웹, 또는 다른 서버의 데이터를 사용해 학습을 할 수 있도록 [PySyft](https://github.com/OpenMined/PySyft), [KotlinSyft](https://github.com/OpenMined/KotlinSyft) 와 같은 라이브러리를 이용해 말단의 기기들용 라이브러리 또한 사용이 가능합니다.

다만, 데모를 따라하시면서 여러 가지 잘 안되는 상황이 발생할 수 있는데 주의점을 알려드리자면 아래와 같습니다.
- Python 3.8 미만 버전과 tensorflow 1.12 이상, tensorflow 2.0 미만 버전을 사용해주세요. (버전 충돌하는 경우가 계속 나오더라구요)
- 한 번 Part1의 Jupyter notebook을 실행하면 이후 실행시 FL process exsists와 같은 error가 뜰 수 있는데 이 때는 name이나 version을 다르게 하고 (이때 kotlinsyft TrainingTask.kt 내에서도 변경) 실행해주시면 됩니다. 또는 docker cache를 삭제함으로써도 가능하다고 하네요. (docker system prune -a, 하지만 이미지도 다 지워지는 것 같으니 참고만 해주세요..)
- 안드로이드 에뮬레이터를 이용하실 경우 SyftConfiguration.kt 파일의 battery check 부분을 false로 바꿔주세요

## 정리 ##

오늘은 Federated learning이라는 모바일 (혹은 말단 웹) 환경에서 딥러닝 모델을 훈련시킬 수 있는 개념에 대하여 확인해보았습니다.
딥러닝 기술이 발전하면서 재미있는 앱들이 많이 나오고 있는데 현실적으로 데이터, Privacy 이슈들이 많이 있다는 것을 알게 되었습니다.
뿐만 아니라, network cost가 큰 상황에서 local computation을 늘림으로써 이를 극복하는 방식도 확인할 수 있었습니다.
시간이 흐르면 흐를 수록 분산형, 연합형 학습이 크게 중요해질 것이라는 생각이 드네요.
감사합니다 :)

