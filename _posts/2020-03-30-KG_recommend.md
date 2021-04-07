---
title: "지식그래프(Knowledge graph)를 이용한 추천시스템"
date: 2020-03-30 12:02:28 -0400
categories: recommender_system knowledge_graph
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Intro ##

GNN, GCN 등 그래프를 활용한 뉴럴 네트워크의 발전과 함께 지식그래프(Knowledge graph)를 이용하는 분야가 확장되고 있습니다.
그 분야 중 한 분야가 바로 추천시스템(Recommender system) 입니다.
추천시스템은 현대 인터넷이 직면한 문제인 '정보의 홍수'를 해결하기 위해 발달했습니다.
예컨대 넷플릭스에는 수천 가지가 넘는 영화들이 존재하고, 소비자는 한땀한땀 자신이 원하는 영화를 찾아내는 것이 쉽지 않습니다.
따라서 사용자가 좋아할만한 영화들을 꼽아 추천해주고 있습니다.

현재 사용되는 추천시스템들은 대부분 두 가지로 나눌 수 있습니다.

<img src="https://miro.medium.com/max/1064/1*mz9tzP1LjPBhmiWXeHyQkQ.png" width="600"/>

User-Item interaction matrix에 기반한 CF-based 방식과 아이템의 속성을 이용해 비슷한 아이템을 추천하는 Content-based 방식입니다.
쉽게 설명하자면 CF-based 방식은, 유저 A와 비슷한 영화 취향을 가지고 있는 B의 시청 목록에 기반해 A에게 추천을 해주는 형태입니다.
반면 Content-based는 A가 좋아한 영화, 예컨대 '기생충'에서 '봉준호'라는 감독 특성을 이용해 '봉준호'의 다른 영화를 추천해주는 형태입니다.

하지만 두 방식은 각각 한계를 가지고 있습니다.
먼저 CF-based 방식의 경우 User-Item interaction 기록이 적으면 적을수록 추천의 성능이 떨어지게 됩니다.
또한 사용자가 처음 서비스를 시작하는 경우 interaction 기록이 아예 없기 때문에 머신 러닝에서 전형적인 cold start problem이 발생할 수 있습니다.
Content-based 방식의 경우 아이템 간의 유사성을 측정하기 위해 제작자가 일일이 feature를 뽑아줘야 하는 불편함이 있습니다.

## 지식그래프 (Knowledge Graph) ##

<img src="https://www.mdpi.com/ijgi/ijgi-08-00254/article_deploy/html/images/ijgi-08-00254-g001.png" width="600"/>

지식그래프는 여러 분야, 굉장히 큰 규모 정보를 나타내기 위한 그래프입니다.
그래프는 노드와 엣지가 있는데 지식그래프에서는 노드는 entity, 엣지는 entity 간의 relation으로 표현됩니다.
말이 조금 어려울 뿐이지 단순 주-술-목 triple들의 집합이라고 생각하면 됩니다.
예컨대 김치찌개는 매운 맛이 있죠.
이걸 지식그래프의 일부로 표현하자면 (김치찌개, has_taste, 매운 맛) 이 되는겁니다.
지식그래프는 heterogeneous network라고도 불립니다.
일반적인 그래프와는 달리 entity(노드)와 relation(엣지)에 여러 가지 type이 정의되기 때문입니다.
하나의 그래프에 김치찌개(food type), 매운 맛(taste type) 이 있는 것처럼 말이죠.

지식그래프를 이용하면 Intro에서 언급했던 CF-based, Content-based 추천시스템의 단점을 보완하여 추천시스템을 개발할 수 있습니다.
눈치 채신분들도 있겠지만 CF-based와 Content-based는 사실 서로 다른 것을 이용하고 있습니다.
CF-based는 오로직 user-item만을, Content-based는 item-attr만을 보고 있죠.
왜 우리는 추천시스템을 만들면서 두 가지 정보를 동시에 고려하지 않았던 걸까요?
지식그래프를 이용해 두 방법을 보완하는 것을 그림으로 나타내면 아래와 같습니다.

<img src="https://dl.acm.org/cms/attachment/473284cc-6bb2-4508-bcdc-bb49b13a26e9/tois3703-32-f01.jpg" width="600"/>

그림을 보면 왼쪽은 유저와 유저가 본 영화(User-Item interaction), 오른쪽은 영화와 영화의 특성들(Item-Attr)이 있는 것을 확인할 수 있습니다.
그리고 맨 오른쪽에는 해당 특성들과 유저가 아직 보지 않은 영화(추천가능한 영화)들이 있습니다.
이처럼 지식그래프를 이용한 추천은 < 유저 - 유저가 Interact한 아이템 - Interact한 아이템의 특성 - 해당 특성을 공유하는 새로운 아이템 > 의 흐름으로 이루어지게 됩니다.

지식그래프를 이용한 추천시스템에는 세 가지의 종류가 있습니다.
각각 지식그래프 임베딩을 이용한 방법, Path를 이용한 방법, GNN을 이용한 방법입니다.
이들에 대해서는 하나 하나 차근차근 살펴보도록 하겠습니다.

## Embedding based method ##

지식그래프 임베딩(knowledge graph embedding)은 지식그래프 내의 entity와 relation을 표현하는 representation learning입니다.
word2vec이 문서 내의 단어들을 벡터화한 것과 같이 지식그래프의 각 element들을 벡터로 표현하는 방법이라고 생각하시면 됩니다.
지식그래프 임베딩에는 굉장히 여러 가지 방법들이 있지만 그 중 가장 간단한 방법 TransE를 소개해드리겠습니다.

<img src="https://www.researchgate.net/profile/Junyan_Qian/publication/320220365/figure/fig1/AS:566530552995840@1512082817303/Simple-illustration-of-TransE.png" width="300"/>

이전에 지식그래프는 triple로 표현된다고 말했습니다.
이는 주-술-목, 혹은 head-relation-tail 이라고 불립니다.
그림에서 h,r,t는 각각 head(entity, node)-relation(edge)-tail(entity, node)를 의미합니다.
즉, TransE는 각각의 element들이 벡터공간 상에서 h+r=t를 만족할 수 있도록 학습이 진행됩니다.
이렇게 학습이 되는 벡터들은 지식그래프(KG) 내부의 의미정보들을 잘 담을 수 있게 됩니다.
예컨대, head와 relation이 주어지면 tail에 올 entity를 알 수 있거나, head와 tail이 주어지면 두 entity 사이의 relation을 예측하는 데에 사용될 수 있습니다.

실제로 지식그래프를 임베딩하는 방식은 굉장히 많습니다.
하지만 본질은 그래프 내부의 구조를 굉장히 잘 학습할 수 있다는 데에 있습니다.
이것이 추천시스템에 오게되면 굉장히 중요한 의미를 가지게 됩니다.
Item과 Attributes들을 이용하여 지식그래프를 만들고, 이를 지식그래프 임베딩을 통하여 학습을 하게되면 Item에 대한 comprehensive representation을 학습할 수 있게 됩니다.
예를 들어 영화 '기생충'이라는 Item entity를 표현하는 벡터를 만들 때는 '감독'(relation)이 '봉준호'(entity)라는 것과, '배우'(relation)에 '박소담'(entity)이 나온다는 것도 포함되기 때문입니다.
이처럼 side information을 포함한, 즉 Content-based의 장점을 살리는 Item vector를 만들 수 있게 되는 것이죠.

이렇게 표현력이 풍부한 Item에 대한 vector를 만들게 되면 해당 vector와 User vector를 이용하여 score function을 계산할 수 있습니다.
예컨대 U vector와 V vector의 inner product $$ \hat{y}_{i,j} = f(u_{i}, v_{j}) $$ 가 될 수 있겠습니다.
그리고 실제 관찰된 interaction data(positive / negative)에 대해 Gradient Descent 방식으로 학습을 할 수 있게 될 것입니다.
단 User vector의 경우 User-Item interaction matrix에 대해 MF 방식을 적용하여 구할 수도 있고, User가 interact했던 item vector들을 잘 조합하여 만들 수도 있을 것입니다.
어찌되었든 본질은 'Embedding 기법을 이용하여 그래프 구조와 의미들을 반영한 Vector를 만들고 Vector들간의 연산을 통해 interaction (1/0)을 예측한다' 입니다.

## Path based method ##

Path based 방법의 경우 굉장히 직관적으로 추천의 경로를 이해할 수 있습니다.
지식그래프는 entity-relation-entity-relation으로 이루어지는 Path를 가지고 있습니다.

<img src="https://www.researchgate.net/profile/Marek_Woda/publication/225743387/figure/fig7/AS:302693768810503@1449179226760/Example-of-course-knowledge-graph-with-weights-and-learning-path-visualization.png" width="600">

이 Path를 이용하면 추천의 정확도와 설명력을 높일 수 있습니다.
기본적으로는 두 가지 방식이 주로 사용되었습니다.
예전에는 Meta-path라는 개념이 도입되어 활용되었습니다.
이 Meta-path는 주로 entity 간의 similarity 들을 계산하기 위해 사용되었습니다.
예를 들어보겠습니다.

<img src="https://tech.ebayinc.com/assets/Uploads/Editor/_resampled/ResizedImageWzEyMDAsNTM2XQ/Screen-Shot-2018-12-02-at-11.59.24-PM.png" width="600">

노래에 관한 User-Item graph가 있다고 할 때에 여러 가지 종류의 Meta-path가 존재할 수 있습니다.
Song-Writer-Song 라던가, Song-Genre-Song 와 같은 형식입니다.
이런 종류의 여러가지 Meta-path들을 사전에 정의해둔다면 실제로 두 개의 다른 entity 간의 유사함을 구할 수 있겠지요.
(쉽게 생각해서 Song-Singer-Song 이 많이 겹칠수록 비슷한 노래일 것입니다)
하지만 위와같이 Meta-path를 이용하면 굉장히 노동집약적인 일이되고 (직접 골라야하므로) '추천'이라는 Task에 직접적으로 반영시킬 수 없다는 단점이 존재합니다.
그래서 보통은 Regularization term으로 이용한다고들 하나, 최근에는 잘 사용되지 않아서 넘어가고자 합니다.

지금은 어떤 시대입니까?
바로 딥러닝의 시대 아니겠습니까?
Path하면 뭡니까, Sequence이죠, Sequence이면 뭡니까 LSTM이 생각나시지 않습니까 :) ??
당연하게도 KG의 Path를 input으로 하는 뉴럴넷 모델들이 등장하기 시작합니다.

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_201784/project_317733/images/x2.png" width="600">

위와같이 knowledge graph 상의 path를 추출하여 LSTM 네트워크의 인풋으로 넣는 것이죠.
해당 네트워크는 Label(1/0) 을 기준으로 학습이 될거고 이후 prediction 시에는 Path input에 따른 prediction score를 output으로 낼 수 있습니다.
위와같은 Path based model의 장점은 결과를 제공하기에 용이하다는 것입니다.
즉 설명이 가능한 추천이라는 큰 장점입니다.
"당신은 이전에 기생충이라는 영화를 좋아했고, 이 기생충 영화의 감독인 봉준호가 감독인 설국열차도 좋아할 것입니다" 와 같은 직관적인 설명이 가능해지는 것입니다.
바로 지식그래프의 relation이 반영되기 때문에 가능한 설명입니다.
(기존 추천 모델은 봉준호라는 entity가 감독인지 출연진인지 구분할 수 없었겠죠)


## GNN based method ##

당연하게도 지식그래프를 이용한 GNN 기반의 추천 모델들도 존재합니다.
하지만 아직 체계화되어있다는 느낌은 없으며 GCN / GAT 기반의 모델을 활용하고 있습니다.
Graph Neural Network는 기본적으로 embedding propagation의 원리 하에 작동하게 됩니다.

<img src="https://www.secmem.org/assets/images/gnn/gnn_1.png" width="600">

그림에서 보는 것과 같이 하나의 노드의 representation을 결정하는 데에 주변 노드 + 자기 자신이 기여를 하게 되는 것이지요.
GNN based 의 방법인 KGCN 모델(https://arxiv.org/abs/1904.12575)을 통해 살펴보겠습니다.

<img src="https://www.programmersought.com/images/470/e3f1243abe52f0dd32225be2c302963e.png" width="600">

<img src="https://raw.githubusercontent.com/hwwang55/KGCN/master/framework.png" width="600">

해당 논문에서서는 Item을 기반으로 지식그래프를 구축합니다.
그림과 같이 한 아이템(여기서는 영화)의 주변에는 아이템의 특성들이 존재할 수 있습니다.
즉 위의 예시를 이용하면 Forrest Gump라는 embedding을 계산할 때 주변의 Drama라는 genre, US라는 country, Tom hanks라는 배우와 Robert라는 감독의 embedding이 영향을 줄 수 있다는 것입니다.
하지만 Forrest Gump 영화에 2단계 attribute인 "Back to the future" 등의 embedding은 영향을 주지 못하는걸까요?
아닙니다.
방금 언급한 Embedding propagation은 그래프 내의 모든 entity에 대해서 진행이 되는데 이 진행과정을 2번, 3번 혹은 n번을 진행하게 되면 n hop 멀리 떨어진 entity도 영향을 줄 수 있습니다.
다시말해, 첫 번째 iteration에서 Drama라는 entity는 Titanic, forrest gump에 영향을 받았을 것이고, 이 새로운 Drama embedding이 두 번째 iteration 때 Forrest gump에 영향을 줍니다.
즉, 두 번째 iteration이 되면 Forrest gump embedding을 결정하는 데에는 Titanic 이라는 2 hop 떨어진 entity 역시 영향을 주게 되는 것이죠.
GNN based recommendation model에서는 이 neighbor들의 중요도에 따라서 영향력을 다르게하는 attention mechanism이 반영됩니다.
즉, Forrest gump embedding 형성에 이웃 중 중요한 녀석들을 더 크게 반영하겠다는 것이지요.
그리고 Knowledge graph에서는 주로 relation을 기반으로 하여 이 중요도를 결정합니다.
어떤 영화에는 감독이 중요하고 어떤 영화에는 배우가 중요할 수 있기 때문에 이것이 학습으로 결정된다고 생각하시면 됩니다.
정말 편리하고 직관적인 해석이 될 수 있습니다.

결론적으로 GNN model의 경우에는 이렇게 embedding propagation 과정을 거쳐 item embedding을 구합니다.
이후는 Embedding based 방법의 경우와 같이 User embedding / Item embedding을 모두 이용하여 score function에 넣게 되죠.
그 score function은 inner product가 될 수도, 다른 형식이 될 수도 있지만요 ^^
어찌되었든 GNN based 방법은 Unified method라는 이름으로도 불릴 수 있습니다.
Embedding을 이용하지만, 이 Embedding이 결국 path based로 전파되기 때문이죠.
따라서 두 모델의 장점을 조합했다고도 할 수 있을 것 같습니다.
