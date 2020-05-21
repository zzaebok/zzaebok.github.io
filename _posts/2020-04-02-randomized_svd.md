---
title: "Fast Randomized SVD"
date: 2020-04-02 12:32:28 -0400
categories: Machine_Learning
---

https://research.fb.com/blog/2014/09/fast-randomized-svd/
https://gregorygundersen.com/blog/2019/01/17/randomized-svd/
sklearn n_iter

## Intro ##
[잠재 의미 분석](https://wikidocs.net/24949)을 사용해 각 document들을 embedding 시키는 작업을 한 적이 있었습니다.
잠재의미분석(LSA)는 Truncated SVD를 이용하여 구현할 수 있었기 때문에, 자연스레 sklearn 패키지의 [함수](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)를 사용했습니다.
사용하던 중에 함수에서 'n_iter' 라는 argument를 발견하였습니다.
생각해보면 SVD 과정에서 iteration이 없는데 저런 이름의 인자가 있다는 것이 신기해서 살펴보기로 하였습니다.
sklearn 의 TruncatedSVD 함수는 randomized SVD 방식을 이용하고 있었습니다.
따라서 저도 공부한 내용을 정리하여 보고자 합니다.


## 순서 ##
1. X = USVt 를 구하고 싶다. (단, 이 X는 dominant한 smaller rank를 가지고 있다는 가정)
2. 그러나 X는 very huge matrix이기 때문에 쉽지 않다.
3. X의 low-rank approximation을 구해서 SVD를 쉽게 구할 수 있으면 좋겠다. (Truncated SVD)
4. random projection matrix p를 X에 곱하여 X의 action을 거의 잡아내는 행렬 z를 만들 수 있다.
5. 이 z를 QR분해하여 얻은 Q는 X의 range를 거의 잡아낼 것이고, Y = QtX는 X의 low-rank approximation이다.
6. 이제 Y의 SVD를 구하면 UySVt를 구할 수 있는데, Y의 S와 Vt는 X의 그것들과 같으므로 U = QUy 이다.
7. 즉 우리는 X의 SVD를 구하지 않고 훨씬 작은 행렬 Y의 SVD를 구한 뒤 Q를 곱하는 연산을 함으로써 빠르게 SVD를 구할 수 있다.
8. 하지만, 위 방식은 vanila 버전이고, 이대로 하면 reconstruction error가 커지게 된다.
9. 커지게 되는 이유는 singular value가 생각만큼 rapid하게 decay하지 않기 때문이다.
10. 이렇게 rapid하게 decay하지 않으면, 임의의 컬럼 수 r을 작게 잡을 수록 reconstruction error가 크게된다.
11. 이를 해결하기 위해서는 두 가지 해결책이 있다.
12. 첫 번째는 oversampling으로서, 단지 r + s 로 만드는 방법 ( r = 5~10 )
13. 그리고 두번째는 power iteration의 방법이 있다.
14. Range(A) = Range(AATA) 이므로, singular valu를 빠르게 decay시켜 dominant singular value와 관련된 information을 뽑을 수 있다.
15. 단, 이 iteration 과정에서 round-off error가 발생한다고 한다. 따라서 매 iteration마다 orthonormalize를 해준다.
16. 이 방법으로는 QR decomposition을 쓰거나 pivoted LU decomposition을 쓴다.
