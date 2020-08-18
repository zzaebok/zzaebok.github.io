---
title: "추천시스템에 사용되는 metrics 정리"
date: 2020-08-18 18:49:28 -0400
categories: recommender_system metrics
---

## Intro ##
기계학습, 딥러닝 분야에서는 모델의 성능을 측정하는 수많은 metric들이 존재합니다.
대표적으로는 정확도(accuracy), AUC score 등이 있습니다.
하지만 이러한 metric들은 추천시스템에 바로 적용하기에는 무리가 있습니다.
왜냐하면 추천시스템은 두 가지 질문에 대답할 수 있는 metric을 찾아야하기 때문입니다.

1. 
2.

그래서 저는 '추천시스템'에서 사용되는 metric에는 어떤 것이 있는지 궁금해졌습니다.
이 글은 [해당 포스트](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)를 읽고 덧대어 정리하는 글입니다.


## metrics ##
추천 문제의 경우 때에 따라 regression 혹은 classification으로 바라볼 수 있습니다.
'사용자가 해당영화에 몇점의 평점을 내릴까'에 대한 문제를 풀면 regression 문제가 되며, '사용자가 해당영화를 좋아할까?'에 대한 문제를 풀면 binary classification이 됩니다.
하지만 대부분의 경우에는 추천 문제를 binary classification문제로 가정하는 것 같습니다.
따라서 기본적으로 classification에서 사용되는 주요 metric들에 대한 개념을 집고가는 것이 좋을 것 같습니다.

- Accuracy

2. Precision

3. Recall

4. F1

5. AUC

## Rank-less recommendation metrics ##
1. Precision@k
2. Recall@k
3. Hit@k

## Rank-aware recommendation metrics ##
1. MRR

2. MAP

3. nDCG
