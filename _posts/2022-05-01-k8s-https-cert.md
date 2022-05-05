---
title: "[WIP] 쿠버네티스 Ingress에 HTTPS 연결 적용하기"
date: 2022-05-01 22:20:28 -0400
categories: kubernetes
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## 보안 ##

나는 개발을 하면서도 보안, 인증과 같은 부분을 잘 알지 못한다.
재미도 없고 누군가 해주겠지라는 생각으로 학부에서 공부도 열심히 하지 않았다.
역시나 할 걸 그랬다...

## HTTPS ##

- why
- 대칭키, 공개키
- 실제

## 쿠버네티스 cert-manager ##

- certificate
- CA
- Issuer
- 


## Nginx Ingress Controller ##

- ingress
  - external access
  - routing
- controllers
  - auth-url, response-headers

## References ##

1. https://opentutorials.org/course/228/4894
2. https://cert-manager.io/docs/tutorials/acme/nginx-ingress/#step-7---deploy-a-tls-ingress-resource
