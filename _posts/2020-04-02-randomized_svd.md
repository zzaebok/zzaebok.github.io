---
title: "Fast Randomized SVD"
date: 2020-04-02 12:32:28 -0400
categories: Machine_Learning
---

https://research.fb.com/blog/2014/09/fast-randomized-svd/
https://gregorygundersen.com/blog/2019/01/17/randomized-svd/
sklearn n_iter

## Intro ##
작성중

## 순서 ##
1. A = QQtA 인 Q를 찾는다. (range of A를 근사하는 low-rank approximation)
2. B = QtA 에 대해 SVD를 하면 더 작은 행렬에 대해 하는 것이기 때문ㅇ
3. Q를 찾는 방법이 Randomized approach.
4. 그러나 이는 singular value가 천천히 decay하는 경우 효과적이지 않다.
5. 따라서 power iteration 과정에서 n_iter가 들어간다.
