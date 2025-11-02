---
title: Preference Optimization
date: 2024-12-11 08:56:28 -0400
categories: machine_learning nlp
redirect_to: https://www.jaebok-lee.com/posts/ko/preference-optimization
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Preference Optimization

대 Large Language Models (LLMs) 의 시대이다.
사람들은 어떻게 하면 더 '나은' 문장을 만들어내는 LLM을 만들지 고민하고 있다.
오늘의 주제는 이 고민의 대답이 될 수 있는 Preference Optimization (혹은 Learning from Human Feedback, Preference Alignment 등으로 불린다) 이 무엇인지, 또 구체적으로는 어떤 방법들이 있는지 살펴보려고 한다.

가장 먼저, 더 '나은' 문장, 혹은 더 '좋은' 대답 이라는건 어떤 것일까?

>Q. 짱구가 우주에서 로봇과 싸우는 이야기를 만들어줘  
>  
>A1. 옛날 옛날에, 짱구는 드넓은 우주로 여행하고 싶은 꿈을 가지고...  
>A2. 짱구가 우주에서 로봇과 싸웠는데 이겼다.

자, 어떤 대답이 과연 더 '좋은', '나은', '선호되는' 대답일까?
두 대답은 모두 질문에 대한 올바른 대답이지만, 대부분의 사람들은 A1을 더 나은 대답이라고 생각할 것이다.
이처럼, 사용자들이 더 원하는, 선호하는 대답을 만들어낼 수 있도록 Language Model 을 학습시키는 방법을 Preference Optimization이라고 한다.
개발자가 원하는 형태의 대답 (재미있는 대답을 만드는 LLM, 번역을 위한 LLM, 뉴스 말투로 대답하는 LLM 등) 을 잘 만들 수 있도록 하는 것이다.

하지만, 한 가지 의문이 있다.
그럼 질문-대답 데이터셋에 대해 A1과 같은 선호되는 대답들만 가지고 Supervised Fine-tuning (SFT) 을 하면 되는 것 아닐까?
그에 대한 대답은 "아니다" 이다.

<p align="center">
<img src="https://imgur.com/ADTHgGj.png" width="500">
</p>

위 그림은 [ORPO](https://arxiv.org/abs/2403.07691)라는 논문에서 제시된 그림이다.
우리의 데이터 셋 중 A1과 같은 선호 대답을 `Chosen`, A2와 같은 비선호 대답을 `Rejected`라고 했을 때, 만일 우리가 `Chosen` 데이터만 가지고 SFT를 한다고 가정해보자.
그림에서 확인할 수 있듯, 우리가 아무리 `Chosen` 데이터로만 모델을 학습시켜도 `Rejected` 데이터들에 대한 모델의 Output Score (Log Probabilities) 또한 함께 높아지는 것을 확인할 수 있다.
`Chosen`과 `Rejected` 모두 우리가 원하는 대답의 Domain이라고 볼 수 있으므로 (짱구 예시에서 A1, A2 모두 우주에서 로봇과 싸우는 내용이기는 한 것 처럼) SFT는 Domain Adaptation의 역할만 하고 있는 뿐, 그 Domain 안에서 비선호 대답 대신 선호 대답을 만들어낼 능력을 배우고 있지 않은 것이다.

<p align="center">
<img src="https://imgur.com/Atl7q7q.png" width="400">
</p>

이 현상은, 우리가 SFT 학습 시에 사용하는 Negative Log Likelihood Loss의 형태를 관찰하면 그 이유를 알 수 있다.
식에서 확인할 수 있듯, Loss는 결국 우리가 맞추어야할 i번째 vocabulary 에 의해 정해질 뿐 그 이외의 vocabulary 에 대해서는 세밀한 조정이 이루어지지 않는다.
즉, 비선호 답변에 대해서는 밀어내는 (거부하는) 학습이 따로 이루어지지 않는다고 볼 수 있는 것이다.

## Proximal Policy Optimization (PPO)

자, 그럼 Preference Optimization이 무엇인지 알게 되었다.
그러면 이렇게 선호 형태의 대답은 생성하면서 비선호 형태의 대답은 생성하지 않으려면 도대체 어떻게 해야할까?
기억하는 사람이 있을 수 있는데, 바로 ChatGPT가 처음 출시되었을 때, OpenAI에서는 Preference Optimization 방법 중 하나인 Proximal Policy Optimization (PPO) 를 사용하여 모델을 학습하였다.

<p align="center">
<img src="https://imgur.com/BSNirTG.png" width="700">
</p>

천천히 그들이 ChatGPT 를 어떻게 학습했었는지 살펴보자.
이 그림에서 Policy는 우리가 학습시키려는 '모델'을 의미한다.

1. 제일 먼저, pre-training이 끝난 GPT 모델에 SFT 학습을 시킨다. pre-trained 모델은 next word prediction으로만 학습된 모델이기 때문에 ChatGPT 처럼 유저 질의, 모델 응답을 만들기 위해 질의-응답 Domain 으로의 학습을 시킨다고 생각하면 된다. (이 학습이 끝난 모델이 위 그림의 빨간색 박스 모델)
2. '선호' 대답이 무엇인지 파악할 수 있는 Reward Model 이라는 것을 학습을 시킨다. 이 모델은 선호 대답과 비선호 대답을 ranking loss를 이용하여 두 대답 사이의 output probabilities 가 멀어지도록 학습하게 된다. 이 때 선호, 비선호 대답의 경우 여러 모델을 통해 같은 프롬프트에 대한 대답을 생성한 후 이를 휴먼 레이블러가 레이블링하게 된다. (초록색 박스 모델)
3. 1번 과정에서 학습된 SFT policy 에 대해, Preference Optimization 학습을 실행한다. Prompt에 대한 output을 생성하면 이 output을 reward model에 입력으로 준 뒤, reward 값을 받아서 이 reward 값이 커지는 방향으로 policy가 fine-tuning되게 된다.

전체적으로 살펴보면, 우리가 원하는 선호 대답을 생성하는 모델의 경우 크게 Supervised Fine-tuning, 그리고 Preference Optimization 두 스테이지를 통해 학습되는 것이다.
그리고, 추가적으로 3번 과정 (Preference Optimization) 에서 고려되는 사항이 있는데, 바로 Kullback Leibler (KL) divergence 를 활용한다는 것이다.
KL Divergence 는 두 확률분포의 차이로, 1번 과정이 끝난 SFT policy 를 Preference Optimization Fine-tuning을 하는 과정에서 원래의 분포 (SFT policy) 에서 크게 벗어나지 않도록 regularization을 한다.
SFT가 끝난 모델은 Domain Adaptation이 끝났다는 점을 상기해보면, Preference Optimization 과정에서 Domain 을 벗어나는 경우의 학습을 제한한다는 의미로 생각할 수 있다.
뿐만 아니라, reward hacking을 막는다는 의미로 해석할 수도 있다.
즉, reward 모델 역시 '학습된' 모델이라는 점에서 취약점이 있을 수 있는데, (예컨대 문장이 길수록 무조건 reward 가 높아진다는 등의) KL divergence term을 추가함으로써 단순히 reward 모델의 취약점만 노리는 행위 (앞선 예시에 대응되는 문장을 그냥 길게만 만드는 행위) 를 방지하는 것이다.

## Direct Preference Optimization (DPO)

이처럼, 앞으로 소개될 다양한 Preference Optimization 방법들은, 각각의 철학에 따라 비선호 형태의 답변보다 선호 형태의 답변을 출력하도록 학습할 것인지를 기술한다고 생각하면 된다. PPO의 경우 Reward 모델을 만들어 선호 형태의 답변 시그널을 생성한 것이다.
하지만 이런 접근 방법에는 큰 문제가 있다.
바로 Reward 모델이 따로 학습되어야 하며, 심지어 Preference Optimization 학습 중에도 사용이 되어야하는 것이다.

이 문제를 해결하기 위해 등장한 새로운 방법이 Direct Preference Optimization (DPO) 이다.
DPO 논문의 제목에 "Your Language Model is Secretly a Reward Model" 라는 표현은 이 떄문에 붙은 것이다.

<p align="center">
<img src="https://imgur.com/vWWOGBX.png" width="600">
</p>

DPO 학습의 Loss를 통해 모델은 '선호 답변에 대한 policy 모델과 ref 모델의 확률 비율'이 '비선호 답변에 대한 policy 모델과 ref 모델의 확률 비율'보다 커지도록 학습된다.
여기서 ref 모델은 SFT Fine-tuning이 끝난, 앞으로 Preference Optimization Fine-tuning이 되기 전의 Freeze 된, 모델을 의미한다. (이전 KL Divergence 에서 $\pi^{SFT}$ 로 표현된 모델)
왜 저러한 Loss가 도출되었는 지는 논문에서 Bradley-Terry preference model 과 optimal policy 에서의 ground-truth reward 를 이용하여 수식으로 증명이 되어있으나, 이 포스트에서는 다루지 않겠다.
저자들은 위의 Objective 가 PPO의 그것과 본질적으로 동일한 것임을 증명했으며, 이러한 '비율의 차이'를 극대화 하는 것이 저자들의 '비선호 답변 대비 선호 답변을 생성시키는' 레시피인 것이다.

<p align="center">
<img src="https://imgur.com/PlqxmjQ.png" width="600">
</p>

또한 저자들은 자신들의 방법 DPO가 PPO보다 효율적임을 증명하였다.

## Odds Ratio Preference Optimization (ORPO)

<p align="center">
<img src="https://imgur.com/VHnwvhG.png" width="600">
</p>

자, ORPO 가 등장했다.
이 학습 방법 역시, 앞에서 본 '비선호 답변 대비 선호 답변의 확률을 높이는' 저자들의 레시피이다.
그들이 (그리고 다른 모든 PO 방법들의 저자들이) 이러한 주장을 하는 것은 다 각자의 사정이 있겠거니 라고 생각하면 된다.

ORPO는 i) 학습을 더 효율적으로 만들고, ii) Preference Optimization 방법과 SFT 과정을 결합헤 1-stage로 학습이 되도록 개선하였다.

<p align="center">
<img src="https://imgur.com/R7sOjyJ.png" width="400">
</p>

학습이 더 효율적이라고 주장하는 부분은 DPO와 달리 비율의 차이 대신 Odds의 차이를 Objective로 사용했기 때문이다 ($ L_{OR} $).
Odds 는 한국어로는 교차비, 혹은 승산비라고 하는데 $ p / (1-p) $ 로 계산되며, 어떤 사건이 일어날 확률을 사건이 일어나지 않을 확률로 나눈 값이다.
만일, 어떤 사건이 일어날 확률이 99%라면 Odds는 0.99 / 0.01 = 99가 된다.
그렇다면 왜 '비율의 차이'를 극대화 하는 대신 'Odds의 차이'를 극대화하는 전략을 사용하였을까?

<p align="center">
<img src="https://imgur.com/yy6vrNR.png" width="500">
</p>

위 그림처럼, 비율과 Odds는 그 범위에서 큰 차이를 보이게 된다.
기본적으로 비율의 분포를 그려보면 0 주변에 많이 뭉쳐있을 뿐더러, $ \beta $와 같은 regularization을 추가하더라도 이 분포가 아주 크게 변하지 않는다.
하지만, Odds의 차이 (정확히는 Odds 비율의 차이) 는 비율의 차이보다 더 넓게 분포가 되는 것을 확인할 수 있다.
이는, 곧 선호 대답과 비선호 대답의 차이를 의미하므로, 학습이 훨씬 더 효과적으로 이루어질 수 있는 것이다.

추가적으로, 저자들은 SFT + Preference Optimization 2-stage 학습 과정을 일원화하여 SFT 과정과 Preference Optimization 과정을 통합했다.
그냥 단순하게 SFT Loss (NLL Loss) 를 더해버림으로써 말이다. ($ L_{SFT} $)


## Contrastive Preference Optimization (CPO)

<p align="center">
<img src="https://imgur.com/TYhtpqE.png" width="600">
</p>

이번 Preference Optimization 모델은 $ \pi_{ref} $ 마저 제거해버렸다.
저자들은 DPO 같은 모델은 Reward 모델을 제거하는 데에는 성공했지만, SFT가 끝난 모델 역시 비율 계산을 위해 (+KL Divergence 계산을 위해) 학습 시에 떠있어야 하기 때문에 학습 비효율성이 초래된다고 주장했다.
하지만 무작정 $ \pi_{ref} $ 를 지워버릴 수는 없다.
이렇게 되면, Bradley-Terry preference model 모델로부터 유도된 비선호 형태의 데이터 대비 선호 형태의 데이터의 학습이 깨지는 것이기 때문이다.

<p align="center">
<img src="https://imgur.com/tty2LOU.png" width="600">
</p>

대신 저자들은, $ \pi_{w} $ 라는 임의의 ideal한 모델 (따라서, $ \pi_{w}(y_{w}|x) $ 를 1로 만들어버리는) 을 가정하였다.
이러한 가정 하의 Objective 를 만족하는 것을 목표로 한 뒤, 이 Objective가 $ \pi_{ref} $을 지워버린 식의 Upper Bound임을 보이고, 이 Upper bound 를 최적화하는 학습을 진행한 것이다.
이를 통해 저자들은 학습 상의 효율성을 증대시키게 되었다. (메모리, 속도 측면)

## Simple Preference Optimization (SPO)

또 다른 하나의 Preference Optimization 방법이다.

<p align="center">
<img src="https://imgur.com/QO6zjzS.png" width="500">
</p>

이젠 익숙하시죠..?
어떤 철학에 따라 이러한 식이 도출되었는지 확인해보자.

<p align="center">
<img src="https://imgur.com/222wgSF.png" width="550">
</p>

사실, 모델이 Inference할 때는 Sequence 길이를 regularization term 으로 이용한다.
저자들은 이것이 Preference Optimization에도 반영해야한다고 생각했고 위 식처럼 만들었다.

<p align="center">
<img src="https://imgur.com/9iuPZSS.png" width="600">
</p>

그래서, 문장의 길이와 상관 없이 log probability를 일정하게 만듦으로서 효과적인 gradient updates를 달성했다고 볼 수 있다.

## Adaptive-Rejection Preference Optimization (ARPO)

지금까지는 일반적인 언어 모델에 대한 Preference Optimization 방법들을 소개했다.
하지만 이 방법은 특히 기계 번역 (Machine Translation) 으로 LLM을 Fine-tuning 할 때 쓸 수 있는 방법으로 소개되었다.

<p align="center">
<img src="https://imgur.com/wKJTA4g.png" width="600">
</p>

즉, `Chosen (winning)`, `Rejected (losing)`로 선정된 두 문장에 대해서 유사도를 구한뒤, 두 문장이 유사하다면 ranking loss의 차이를 일부 줄여주는 방식을 택한 것이다.

그들의 사정은 무엇이었을까?
기계 번역에서의 선호 - 비선호 데이터의 예시를 통해 살펴보자.

>**Original**: 2004 Olympic silver medallist Amir Khan said, "Deep down I think women shouldn’t fight. That’s my opinion.”  
>  
>**Chosen**: 2004년 올림픽 은메달리스트인 Amir Khan은 "내 생각에 여성들이 싸우면 안 된다고 생각해요. 그게 제 의견입니다."라고 말했다  
>**Rejected**: 2004년 올림픽 은메달리스트인 아미르 칸은 이렇게 말했다, "심사숙고해서 여성들은 싸워서는 안된다고 생각합니다. 그냥 제 의견입니다.”

기계 번역 Task 에서의 선호 - 비선호 데이터는 실제로 굉장히 많은 부분이 일치하는 형태를 띄게 된다.
번역문의 전부가 틀린다기 보다는 일부 구, 표현, 단어의 차이가 Reward 로 결정되기 때문이다.
따라서, 두 문장이 거의 비슷하다고 할 때 DPO, CPO와 같은 일반적인 Preference Optimization Loss를 사용하게 되면 모델에게 "어라? 얘넨 거의 비슷한데 왜 서로 멀어지게 하지?" (문과적 해석입니다.) 와 같이 혼란을 줄 수 있기 때문이다.

마지막으로 강조하지만, 각 Preference Optimization 방법들은 개별의 사정이 있는 것이다.

## 정리

Preference Optimization의 의미와 다양한 방법론들에 대해 조사해 보았다.
각 상황에 따른 사정이 생기고, 이 사정을 해결할 수 있는 레시피들이 개발되는 것이다.
오늘 이 포스트에서 다룬 방법 말고도 정말 수많은 논문들이 존재한다.

<p align="center">
<img src="https://imgur.com/3Kn41S0.png" width="600">
</p>

그럼 이걸 다 구현해서 써야하냐?
아니다.
`huggingface`에는 [`trl`](https://github.com/huggingface/trl)이라는 라이브러리가 있고, 이 곳에서 대부분의 Preference Optimization을 위한 Trainer class가 정의되어 있다.


## References
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model
](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://aclanthology.org/2024.emnlp-main.626/)
- [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation
](https://arxiv.org/abs/2401.08417)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [X-ALMA: Plug & Play Modules and Adaptive Rejection for Quality Translation at Scale](https://arxiv.org/abs/2410.03115)

