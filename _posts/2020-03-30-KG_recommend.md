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
Content-based 방식의 경우 아이템 간의 유사성을 측정하기 위해 제작자가 일일이 feature를 뽑아줘야 하는 굉장히 노동집약적인 일이 필요합니다.

## 지식그래프 (Knowledge Graph) ##
![kg_recommend](https://storage.googleapis.com/groundai-web-prod/media/users/user_306803/project_410582/images/x1.png.344x181_q75_crop.png)

## 

## 목차 ##
