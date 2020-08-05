---
title: "지식그래프(Knowledge graph)를 이용한 추천시스템"
date: 2020-03-30 12:02:28 -0400
categories: knowledge_graph recommender_system
---

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

하지만, 지식그래프를 이용한 추천시스템에는 세 가지의 종류가 있습니다.
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

## GNN based method ##
