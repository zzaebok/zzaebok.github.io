---
title: "Helm upgrade 할 때 latest 이미지가 사용되지 않는 경우 해결법"
date: 2021-12-16 20:03:28 -0400
categories: kubernetes
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## 문제 ##

요즘 Helm을 배우고 있다.
필요한 service, deployment, ingress 같은 k8s 자원들을 묶어서 릴리즈할 수 있어서 매우 편하다고 느끼는 중이다.
하지만 최근들어 (내 생각에는) 이상한 현상을 발견했고, 이를 꼼수로 우회하는 방법을 찾아서 공유하고자 한다.
따라서 올바른 해답을 알고계시는 분이 이 글을 읽으신다면, 꼭 해결법을 알려주세요 ㅠㅠ.

내가 맞딱드린 문제는, `helm upgrade` 명령어가 `latest` 도커 이미지를 가져오지 못하는 것이었다.
정확히는 가져오지 못한다기보다, 업그레이드하지 못한다는 것이다.
만일 내가 아래와 같은 디플로이먼트를 배포했다고 가정해보자

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: example-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      name: example
  template:
    metadata:
      labels:
        name: example
    spec:
      containers:
      - name: sample
        image: sample-image
        imagePullPolicy: Always
```

위 디플로이먼트는 `sample-image:latest` 이미지를 `Always` Pull 하는 정책으로 컨테이너를 생성하게 된다.
만일 내가 이미지를 수정한 뒤 그대로 Push 하여 `sample-image:latest`가 변경되었다고 가정하자.
나는 당연히 `helm upgrade` 명령어를 실행한다면, 컨테이너가 새롭게 바뀐 latest 이미지를 가지고 재생성 될 것이라고 생각했다.
하지만, `helm upgrade` 이후에 새로운 이미지를 가져오기는 커녕 아무 변화도 없었다.
혹시나 하는 마음에 `Chart.yaml` 내의 여러 버전들을 바꾸어도 봤지만 소용이 없었다.
그래서 `helm uninstall` 과 `helm install`을 반복하는 무지성 실험을 하다가 결국 `helm upgrade`를 하는 방법을 찾아 나서게 되었다.

## 해결방법 ##

내가 찾아낸 해결 방법 자체는 굉장히 간단하다.
매번 `helm upgrade`를 할 때마다 저 디플로이먼트 yaml파일에 변화를 주면 된다.
그렇다면 이미지 Pull 정책에 따라 매번 새롭게 변화된 latest 이미지를 가지고 오게 될 것이다.
어떻게 배포 때마다 yaml 파일에 변화를 줄 수 있을까?
일일이 손으로 label을 바꿀 수도 있다.
하지만 이보다는 수월한 방법은 바로 "시간"을 이용하는 것이다.
우리가 배포하는 시간이 정확히 동일한 시간일 수 없으니까 말이다.

{% raw %}
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: example-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      name: example
  template:
    metadata:
      labels:
        name: example
        ################ 여기 #################
        date: "{{ now | unixEpoch }}"
        ######################################
    spec:
      containers:
      - name: sample
        image: sample-image
        imagePullPolicy: Always
```
{% endraw %}

위 yaml파일에서 label에 date를 추가해주었다.
unixEpoch 은 [유닉스 시간](https://ko.wikipedia.org/wiki/%EC%9C%A0%EB%8B%89%EC%8A%A4_%EC%8B%9C%EA%B0%84) 으로서, 1970년 1월 1일 00:00 (UTC) 부터의 경과 시간을 초로 환산하여 정수로 나타낸 것이라고 한다.
따라서 우리가 `helm upgrade`를 통하여 릴리즈를 변경할 때마다 해당 yaml파일의 내용이 변경되게 되고, 따라서 매번 새로운 latest 이미지를 가지고 오게 된다.

## References ##

1. https://github.com/helm/helm/issues/5696