---
title: "지식그래프를 통한 설명력있는 추천"
date: 2020-04-02 12:32:28 -0400
categories: recommender_system knowledge_graph
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Intro ##
최근들어 기계학습을 이용한 Recommendation이 각광받고 있다.
사용자의 니즈를 파악하고 원하는 것을 제공하는 서비스는 매력적일 수밖에 없다.
넷플릭스를 필두로 많은 기업들이 기계학습을 통해 사용자를 분석하고 취향저격 컨텐츠를 만들고자 노력하고 있다.

Recommendation을 위한 기계학습 방법에는 대표적으로 matrix factorization이 있으며, linear regression이나 neural netowrk를 활용하는 곳도 있다.
하지만, 위 방법들을 통한 recommendation은 반쪽짜리일 뿐이다.
왜냐하면 사용자도, 개발자도 무슨 이유로 사용자에게 A라는 아이템을 추천했는 지 알 수 없기 때문이다.
그저 해당 유저와 아이템 사이의 score가 높았을 뿐이다.

## Explainable reasoning ##
이를 극복하고자 한 사례가 바로 knowledge graph를 이용한 것이다.
[Explainable reasoning over knowledge graph for recommendation](https://arxiv.org/pdf/1811.04540.pdf)의 저자들은 knowledge graph의 path를 통해 설명력 있는 reasoning이 가능하다고 주장한다.
가령 당신이 영화 기생충을 좋아한다고 하자.
당신이 영화 '기생충'을 좋아하는 바로 그 이유는 무엇이 될 수 있을까?
영화 배우 박소담(사랑합니다), 송강호가 좋을 수도 있다.
또는 감독인 봉준호의 작품 제작 방식이 좋았을 수도, 비슷한 취향의 친구가 재밌다고하여 좋아할 수도 있다.
이러한 설명들은 모두 '기생충'이라는 영화에 대한 knowledge(지식)이다.
따라서 영화에 대한 knowledge graph를 구축한다면 이를 통한 설명력 있는 추천이 가능해지게 된다.
사용자에게 띡 '너 기생충 좋아할걸?' 보다는 '좋아하는 배우 박소담이 나오는 영화입니다', '비슷한 취향의 친구들이 즐겨 관람한 영화입니다' 와 같이 충분한 설명력을 제공하는 것이 훨씬 매력적이다.

## Knowledge Graph ##
지식그래프 (Knowledge graph)는 node와 edge로 이루어진 말그대로 '그래프'이다.
이 그래프를 지식그래프라고 부르는 이유는 node가 'entity' (예에서 유저, 배우, 영화 등) edge가 'relation' (배우가 누구인지?, 누구의 영화인지?)을 나타내기 때문이다.
보통 지식그래프는 트리플로 표현할 수 있다고 하는데 (h,r,t) head, relation, tail의 꼴이다.
예를들어 (유저A, interact, 기생충), (봉준호, isDirectorOf, 기생충)와 같이 표현할 수 있다.

![kg](https://tech.ebayinc.com/assets/Uploads/Editor/_resampled/ResizedImageWzEyMDAsNTM2XQ/Screen-Shot-2018-12-02-at-11.59.24-PM.png)

위 그림처럼 지식그래프의 일부를 볼 때 Alice라는 유저가 Castle on the Hill 이라는 노래를 선호하게 되는 path는 무엇이 있을까?

p1 = (Alice -> Interact -> Shape of You -> Is Song Of -> ÷ -> Contain Song -> Castle on the Hill)

p2 = (Alice -> Interact -> Shape of You -> Sung by -> Ed sheeran -> IsSingerOf -> Castle on the Hill)

p3 = (Alice -> Interact -> Shape of You -> InteractedBy -> Tony -> Interact -> Castle on the Hill)

각각 p1은 ÷의 수록곡, p2는 가수 Ed sheeran, p3는 Tony라는 유저를 통한 path들이다.
따라서 우리가 유저 Alice가 Castle on the Hill이라는 노래를 좋아할 것인지에 대한 score는 위 path들이 얼마나 '그럴듯 한가'에 달려 있다.


## KPRN ##
저자들은 자신들의 모델 이름을 KPRN이라고 지었다.
Knowledge-aware Path Recurrent Network라는 의미이고, 이름이 직관적으로 내포하는 것처럼 knowledge graph의 path에 RNN 모델을 적용한 것이다.

![kprn](https://storage.googleapis.com/groundai-web-prod/media/users/user_201784/project_317733/images/x2.png)

모델링이 매우 단순하다.
각 entity, entity type, relation을 concat 시킨 뒤 LSTM layer를 이용해 sequence적인 의미와 그럴듯함을 계산해낸다.
단, path가 길어질 수록 결과에 대한 해석이 힘들어지기 때문에 max length는 임의로 잡아주는 것이 좋다.
특이한 점은 item에 대한 score를 내릴 때 weighted pooling layer를 추가했다는 것이다.
즉, 하나의 item에 대하여 여러 path가 존재하고 각 path별로 score가 있을 텐데 이 path별 score에 pooling을 적용했다.
이는 각 path별로 어떤 path가 중요하게 작용하였는 지를 결정하기 위함이다.
다른말로 내가 기생충을 좋아할 수 있는 path는 여러 가지가 있지만, 배우 박소담(사랑합니다) 때문에 봤다는 것을 '설명력'으로 사용할 수 있게 하기 위함이다.

어찌되었든 본 논문에서는 영화, 그리고 음악 dataset에 대하여 KPRN 모델을 실험하였고 Performance 역시 matrix factorization이나 CF-based recommendation을 뛰어넘는 것을 확인하였다.
실력도 좋은데 설명력까지 제공할 수 있다라는 것은 굉장히 큰 의미가 있다고 생각한다.

## Conclusion ##
날이 갈수록 Knowledge graph에 대한, 아니 그래프 자체에 대한 중요성이 커지고 있다.
AI, Machine learning과 관련되어 있는 학회에서도 'Graph'라는 단어의 등장이 급속하게 많아지고 있다고 한다.
결국 사람을 흉내내고 사람의 Reasoning을 따라가기 위해서는 방대한 지식과 그것을 효율적으로 나타낼 수 있는 방법이 필요하다.
이번 논문을 통해 어떻게 지식그래프가 활용 되는지, 기존의 Rule base method와는 어떤 것이 다른 지 알아볼 수 있었다.
path를 통한 설명력이라는 것이 굉장히 직관적이고 이해하기 쉬운 만큼 관련된 다른 논문(KGAT과 같은..)들도 읽고 리뷰를 남겨야겠다.
