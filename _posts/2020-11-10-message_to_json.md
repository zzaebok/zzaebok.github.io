---
title: Grpc protocol buffer message to JSON 변환
date: 2020-11-10 19:18:28.000000000 -04:00
categories: backend
permalink: "/grpc/message_to_json/"
redirect_to: https://www.jaebok-lee.com/posts/ko/message-to-json
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## 문제 상황 ##
grpc 프로토콜을 이용한 서비스에서 서비스의 응답은 시간에 따라 변할 수 있다.
예컨대 현재 유저를 모두 가지고 오는 서비스에서의 응답(현재 유저)은 매 번 변하는 값이다.
만일, 우리가 해당 서비스의 응답을 이용해서 여러 개의 모델을 테스트한다고 가정하자.
유저별로 영화 추천을 해준다든지, 유저에 따른 다른 output을 만들어내야하는 상황이다.
이 때 두 개(두 개라고 가정하자)의 모델에 대하여 아무리 요이 땅! 하고 서비스의 응답을 받아오더라도, 받아온 결과가 달라질 수 있다.

이를 해결하기 위해서는 두 개의 모델이 하나의 서비스 응답을 이용해야 한다.
물론 코드를 잘 짜서 하나의 서비스 응답을 이용해서 두 개의 모델이 이를 이용해 평가/추천 등의 업무를 수행할 수 있다.
하지만, 테스트와 같이 단위를 통일하여 모델을 평가하고 싶을 때와 같은 경우 / 두 개의 모델이 각각 서비스 응답을 받게 설계된 경우에는 곤란해진다.
이 때 가장 떠오르는 방법은 서비스의 응답을 한 번 받아서 '파일'로 저장하고, 평가시에 두 모델이 모두 이 파일을 인풋으로 받아서 사용하는 것이다.

하지만, grpc protocol buffer는 직렬화되어서 통신이되므로 이 응답을 어떻게 저장 / 복원해야하는 지 막막하다.
pickle을 이용해 바이너리로 저장을 시도했지만 '당연하게도' 불가능했다.

고민하던 중 protobuf 내에 [json_format](https://googleapis.dev/python/protobuf/latest/google/protobuf/json_format.html) 이라는 라이브러리를 이용하면 protobuf message를 json형식으로 저장할 수 있다는 것을 알게 되었다.
하지만, 내부에 있는 MessageToJson 함수를 어떻게 사용하는지 감을 못잡다가 결국 알게되어 이를 정리하고자 한다.

## 해결법 ##
먼저, json_format에서는 Message를 Json 형태로 만들어주는 `MessageToJson()`이라는 함수가 있다.
이 함수의 리턴 값은 아래와 같은데,
> A string containing the JSON formatted protocol buffer message.

도대체 JSON 포멧화된 proto buffer message를 포함한 문자열 이라는게 무엇인지 이해가 되지 않았다.
이는 말그대로 proto buffer message를 JSON처럼 만든 하나의 큰 문자열을 의미한다.
즉, "{"name": "zzaebok"}" 과 같이 json이 string으로 감싸진 형태라는 것이다.

따라서 이를 파일로 저장하려면 json.dump 등을 이용하는 것이 아니라 파일 입출력의 write를 이용하면 된다.

예시를 들어보자, 이 글이 필요한 사람들은 grpc를 이용할 줄 안다고 가정하고 [해당 예제](https://grpc.io/docs/languages/python/quickstart/) 의 message들을 이용하여 설명하겠다.

service Greeter를 이용하여 HelloRequest를 날리면 HelloReply 응답을 받을 수 있다.
이 때 응답은 아마
```python
response = stub.SayHello(helloworld_pb2.HelloRequest())
```
와 같은 형태로 받게 될 것이다.

그렇다면 이 response는 helloworld_pb2.HelloResponse 타입일텐데 이 결과값을 그럼 어떻게 파일에다가 저장하는가?
아래와 같이 하면 된다.

```python
from google.protobuf.json_format import MessageToJson
with open('anywhere.json', 'w') as f:
    f.write(MessageToJson(response))
```
나로서는 MessageToJson이라는 함수명을 바탕으로 json obj 형태의 output이 나올 줄 알고 json 라이브러리를 이용했던터라 시간낭비를 했다.
json파일을 만들고 그 안에 Json 포멧화된 message 문자열을 써버리면 되는 것이다.

반대로 해당 파일을 통해 똑같이 response를 복원하기 위해서는 
```python
from google.protobuf.json_format import Parse
with open('anywhere.json', 'r') as f:
    response = Parse(f.read(), helloworld_pb2.HelloResponse())
```
위의 형태로 복원하면 된다.
파일 내의 문자열 (json형식)을 읽어 Parse 함수에 넣어주면 된다.
여기서 주의할 점은 해당 JSON 포멧의 문자열이 **어떤 메세지 형태**인지를 추가 인자로 넣어줘야한다는 점이다.
여기서 우리가 복원할 메세지는 helloworld_pb2.HelloResponse 타입이므로 해당 메세지를 넣어주면 된다.
또 하나 주의할 점은 helloworld_pb2.HelloResponse가 아니라 helloworld_pb2.HelloResponse() 처럼 괄호를 빼면 안된다는 점이다.

이렇게 protocol buffer message를 json형식으로 바꾸어 파일로 저장하는 방법에 대해 알아보았다.
물론 테스트를 이렇게 안하는 사람도 있고 더 스마트하게 하는 방법이 있을 거라고 생각한다.
하지만 내가 현재로 생각나는 방법은 파일로 저장하여 어떤 시점에의 응답을 기록하는 방법이었다.
MessageToJson / MessageToDic 같은 함수들을 찾을 수 있었지만 파일로 저장하고 복원하는 것에 대한 정리가 되어 있지 않아 이렇게 기록으로 남겨 둔다.
도움이 되었길 바라며...
