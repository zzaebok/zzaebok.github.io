---
title: 쿠버네티스 Ingress에 HTTPS 연결 적용하기
date: 2022-05-01 22:20:28.000000000 -04:00
categories: kubernetes
redirect_to: https://jaebok-lee.com/posts/k8s-https-cert
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
머리가 나쁘면 몸이 고생한다나

## HTTPS ##

HTTPS 는 HTTP 프로토콜에 보안이 강화된 것이다.
HTTP는 기본적으로 암호화가 되어있지 않기 때문에 주고 받는 데이터 내용이 쉽게 유출될 수 있다고 한다.
네이버나 구글 등 로그인 화면에 가면 주소창 맨 왼쪽에 자물쇠모양이 생기는 것을 확인할 수 있는데, 이처럼 보안이 중요한 내용을 전송할 때 HTTPS 프로토콜이 사용된다.

먼저, '암호화'라는 것이 어떻게 이루어지는지 살펴보자.
가장 간단하게는 '대칭키 암호화'라는 것이 있다.
이는, 암호화와 복호화 시에 동일한 키를 이용하는 방법으로 가장 간단하다고 볼 수 있다.
하지만 대칭키를 이용하면 이 키를 주고 받는 것(전송)이 문제가 되는데, 중간에 어떻게라도 키를 가로챈다면 전송 내용 역시 그대로 빼앗길 수 있다.
이러한 문제점을 해결하고자 나온 것이 '공개키 암호화'방식이다.
공개키 암호화 방식에서는 두 개의 키를 이용하여 암호화를 하게 된다.
만일 1번 키를 이용하여 암호화를 하면 2번 키로 복호화할 수 있고, 2번 키로 암호화를 하면 1번 키로 복호화를 할 수 있다.
예를 들어보자, A서버가 자신과 통신하고 싶은 B서버에게 1번키를 주며 말한다 "이걸로 암호화해서 보내".
그러면, B서버는 1번키를 이용해 전송 내용을 암호화 하고 A서버에게 보낸다.
A서버는 2번 키를 가지고 있기 때문에 B서버가 보낸 암호화된 내용을 복호화하여 확인할 수 있다.
하지만, 해커는 2번 키를 가지고 있지 않기 때문에 A서버가 B서버에게 알려준 1번 키를 훔쳤다고 하더라도 내용을 복호화할 수 없다.
이 때 A가 공개하는 1번 키를 "공개키(public key)", 자신만 가지고 있는 2번 키를 "비밀키(private key)"라고 한다.

HTTPS 프로토콜에서는, 'SSL(TLS) 인증서'라는 것을 서버에서 준비하게 된다.
클라이언트가 서버에 접속했을 때 서버는 이 인증서를 건내며, "봐봐, 나 신뢰할 수 있는 놈이야"라고 말한다.
그럼, 클라이언트는 "아! 이녀석은 내가 찾던 녀석이구나!"하고 신뢰할 수 있게 되는 것이다.

<img src="https://imgur.com/PRByWgF.png" width=800>

위 그림을 통해 HTTPS의 동작을 살펴보면,
1. 유저가 어떤 서버에 접속을 하려고 하면 (예를 들면 주소창에 www.naver.com을 입력),
2. DNS 서버에서 해당 서버의 IP 주소를 알아내어 
3. 서버에 접속을 한다. 
4. 이 때 사용자의 브라우저는 SSL(TLS) 인증서를 보여달라고 하고 
5. 서버는 유효한 SSL(TLS) 인증서를 보여주어 "내가 니가 찾던 그 서버야" 라고 말한 뒤 
6. 통신이 이루어지게 된다.

그런데 문제가 하나 있다.
서버가 인증서를 보여주며 "나 신뢰할 수 있는 놈이야!" 라고 말할 때, 진짜로 신뢰할 수 있을까?
해커가 그냥 인증서를 아무렇게 만들어서 믿으라고 하는 것일 수도 있다.
이를 방지하기 위하여 'CA(Certificate Authorities)'라는 인증 기관들이 존재한다.
이는 공인된 기관들이며, 각 브라우저는 이 믿을만한 CA 목록을 가지고 있어서, 전달받은 SSL(TLS) 인증서가 어디로부터 발급된 것인지를 확인함으로써 서버가 공인된 것임을 확인할 수 있다. (즉, 우리가 서버의 입장이라면 이 CA들로부터 인증서를 발급/구매 하여야 하는 것이다)

만일, 해커가 이 인증서조차 CA로부터 발급된 것처럼 꾸밀 수도 있지 않을까?
불가능하다.
왜냐하면, 이 인증서는 CA가 자신의 '비밀키'로 암호화 해놓았기 때문이다.
따라서 우리의 브라우저는 이미 공개된 CA의 공개키를 이용하여 이 인증서를 복호화하였을 때 성공한다면, 이 인증서가 실제 믿을만한 CA의 비밀키로 암호화된 것임을 확인할 수 있다.

이 인증서를 복호화하면 나오는 정보 중에는, 해당 서버의 공개키와 공개키의 암호화 방법이 들어있다.
따라서 서버로부터 인증서를 받은 뒤에는 어떤 방법으로 암호화 통신을 할 수 있는 지 알 수 있고 보내려는 정보를 서버의 공개키를 이용하여 암호화한 뒤 전송하면 된다. (그림에서 6번)


## 쿠버네티스 cert-manager ##

이렇게 내가 개발하는 서버가 HTTPS 연결이 가능하도록 하려면 SSL(TLS) 인증서라는 것을 준비해놓아야한다.
쿠버네티스에서 이렇게 인증서 발급, 사용, 갱신 등을 하는 리소스들을 사용하도록 도와주는 것이 바로 [cert-manager](https://cert-manager.io/docs/) 이다.
이 글에서는 `cert-manager`를 `helm`을 통해 이용하여 NGINX ingress 를 [안전하게 세팅](https://cert-manager.io/docs/tutorials/acme/nginx-ingress/#step-2---deploy-the-nginx-ingress-controller)하는 방법을 따라해볼 것이다.
우리가 이미 라우팅하려는 서비스는 `hello` 라는 이름으로 배포되어 있고, DNS name은 `example.com`으로 가정하겠다.

먼저, `NGINX Ingress Controller`를 배포한다.
Ingress Controller는 `ingress`라는 쿠버네티스 명세를 이용하여, HTTP(S) 트래픽에 대한 접근을 통제할 수 있다.
마이크로서비스를 상황이라면, `ingress`에 지정된 path 에 따라서 각 서비스로 트래픽을 라우팅해주는 것이다.
예컨대, `example.com/user` 는 유저 서비스 pod로, `example.com/item` 는 아이템 서비스 pod로 트래픽을 라우팅해주는 것이다.

```sh
$ helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
$ helm repo update
# nginx라는 이름으로 controller 설치
$ helm install nginx ingress-nginx/ingress-nginx
```

[`values.yaml`](https://github.com/kubernetes/ingress-nginx/blob/main/charts/ingress-nginx/values.yaml) 파일에서 필요한 부분을 변경하면 된다.
특히 HTTPS 서버의 경우 도메인에 SSL(TLS) 인증서가 발급되는데, 따라서 고정 IP가 사용되게 되고 `loadBalancerIP` 부분을 수정해주면 된다.

그 다음으로 `cert-manager`를 배포한다.

```sh
$ helm repo add jetstack https://charts.jetstack.io
$ helm repo update
# cert-manager라는 이름으로 cert-manager 설치
$ helm install cert-manager jetstack/cert-manager --namespace cert-manager --create-namespace --version v1.8.0
```

cert-manager는 쿠버네티스 클러스터 내에서 인증기관에게 인증서를 만들어 줄 것을 요청하고, 인증기관의 challenge에 응답하는 역할을 수행한다.
challenge는 쉽게 말해, 인증기관에서 당신이 DNS name의 주인인지를 파악하는 과정이라고 생각하면 된다.
`example.com`에 대한 SSL(TLS) 인증서를 발급받기 위해, `example.com` 도메인이 내 것인지를 확인하는 과정이다.
대표적으로 HTTP-01 타입의 challenge가 있는데, 인증기관에서 어떤 토큰을 주면 우리는 이를 `http://<YOUR_DOMAIN>/.well-known/acme-challenge/<TOKEN>` url에서 가져갈 수 있게 파일을 업로드하는 형식이다.
인증기관이 성공적으로 해당 파일에 접근할 수 있다면, 해당 도메인이 우리 것이라는 것을 확인하는 식이다.

이 다음에는 어떤 인증기관에다가 인증서 생성을 요청해야할지 정의하면 된다.
`cert-manager`는 내부적으로 CRD 리소스인 `Issuer`를 사용하는데, 이 `Issuer`리소스가 바로 '어떻게 `cert-manager`가 TLS 인증서를 요청할지'를 정의하게 된다.

```yaml
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
 name: letsencrypt-production
spec:
 acme:
   # ACME 서버 URL
   server: https://acme-v02.api.letsencrypt.org/directory
   # ACMD 등록을 위한 이메일 주소
   email: user@naver.com
   # ACME 계정 비밀키를 저장할 Secret 이름
   privateKeySecretRef:
     name: letsencrypt-production
   solvers:
   - http01:
       ingress:
         class:  nginx
```

여기서 ACME 란 Automated Certificate Management Environment를 의미하며, 자동으로 X.509 인증서를 발급할 때 사용하는 프로토콜이다.
해당 `Issuer` 리소스는 우리가 `Let's Encrypt`라는 인증기관의 ACME 서버를 이용한 인증서를 요청할 것이라는 내용을 담고 있다.
`email`은 `Let's Encrypt` 계정 등록을 위해 사용되고, 인증서의 만료 등과 관련되어 연락수단이 된다.
그리고 `privateKeySecretRef`는 그 계정의 비밀키가 저장될 곳이다.
계정의 비밀키는 위에서 말한 challenge 단계에서 사용되는데, `Let's Encrypt`에서 준 임의의 TOKEN(nonce)을 이 private key를 이용해 서명하여 도메인 아래 특정 url에 올려둔다.
그러면 `Let's Encrypt`는 해당 url을 통해 그 서명된 TOKEN을 가져가 이번에는 계정의 공개키를 이용하여 서명을 확인함으로써, 계정이 실제로 해당 도메인을 가지고 있는지 확인하게 된다.
이후 우리가 해당 도메인에 SSL(TLS) 인증서 발급을 요청하면, 우리가 주인임을 확인했음으로 인증서 발급을 진행하게 되는 것이다.

마지막으로 SSL(TLS) 연결을 위한 `ingress` 리소스를 배포한다.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myingress
  annotations:
    kubernetes.io/ingress.class: "nginx"    
    cert-manager.io/issuer: "letsencrypt-production"

spec:
  tls:
  - hosts:
    - example.com
    secretName: example-tls
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hello
            port:
              number: 80
```

`metadata.annotations`를 통해 인증서 issuer를 기록해준다.
그리고 tls 연결 설정은 `spec.tls`에서 할 수 있는데, 도메인 명과 발급된 certificate 내용이 담길 secret명을 적어주면 된다.

## 마무리 ##

이렇게 이악물고 모른척 넘어가려고 했던 HTTPS 와 인증서, 그리고 이를 쿠버네티스에 적용하는 법을 정리해보았다.
그래도 다행히 reference에 달아둔 생활코딩님과 `cert-manager` 문서, `Let's Encrypt` 문서를 통해 비교적 차근차근 개념을 잡아갈 수 있었다.
물론 부족한 부분이 아직 매우매우 많지만.. 미래의 내가 알아서 할 거다^^ 끝!

## References ##

1. https://opentutorials.org/course/228/4894
2. https://cert-manager.io/docs/tutorials/acme/nginx-ingress/#step-7---deploy-a-tls-ingress-resource
3. https://letsencrypt.org/how-it-works/
