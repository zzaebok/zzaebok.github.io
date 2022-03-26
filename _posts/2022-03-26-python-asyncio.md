---
title: "[Python] 파이썬 비동기 프로그래밍 asyncio 겉핥기"
date: 2022-03-26 19:01:28 -0400
categories: python
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## 비동기 프로그래밍 ##

머신러닝 위주의 업무만 하다보면, '개발자'라는 직업을 가지고 있지만 실제로는 개발에 관해 모르는 부분이 많이 생긴다.
이 중 가장 대표적인 것이 나에게는 비동기 프로그래밍이었다.
'비동기로 웹 API를 호출한다', '비동기로 데이터베이스를 연결한다'와 같은 말들을 들을 때면 항상 "도대체 비동기가 뭐지...?" 라는 생각을 하곤 했다.
그래서 이 기회에 비동기 프로그래밍을, 특히 파이썬의 `asyncio` 라이브러리 위주로 정리해보고자 한다.
특히 내 입장에서 '언제' 쓰면 좋을지, '어떻게' 쓰면 좋을지에 대한 기록이다.

동기 vs 비동기 프로그래밍에 대한 [재미있는 글](https://fastapi.tiangolo.com/async/#asynchronous-code)이 있다.
우리가 친구와 함께 햄버거 매장에 간다고 해보자.
먼저 '동기' 매장은 주문을 접수하고 햄버거를 받기 전까지 매대 앞에서 기다려야한다.
자칫 한 눈을 판다면 햄버거가 나온 뒤 모르는 사람이 자신의 주문인 척, 햄버거를 수령해가버릴 수도 있다.
하지만 '비동기' 매장은 주문을 접수하면 번호표를 나눠준다.
나는 이 번호표를 가지고 자리에 앉아 친구와 재미있게 이야기를 나누고 있으면 된다.
이 글에서 눈치 챌 수 있듯이, 동기 프로그래밍에서는 프로세스가 어떤 작업을 기다리는 동안 아무것도 하지 않고 기다린다.
반드시 앞의 작업이 끝나야 뒤에 것을 하게 설계된 것이다.
하지만 비동기 프로그래밍에서는 어떤 작업을 기다리는 동안에도 프로세스가 다른 일 (위의 예시에서는 친구와 떠드는)을 할 수 있다.

비동기 프로그래밍이 효과적인 경우는 I/O bounded 작업을 하는 경우이다.
데이터베이스에 접근한다든가, 네트워크 통신을 해야하는 경우는 현재 나의 프로세스와 상관 없이 결과를 받기 까지 오랜 시간이 걸린다.
따라서 그 시간동안 놀기보다는 또 다른 일을 시키는 것이 효율적이라고 할 수 있다.

## 파이썬, asyncio ##

그렇다면 파이썬에서는 어떻게 비동기 프로그래밍을 할 수 있을까?
파이썬엔 'event loop' 라는 친구가 있다.
이 친구는 기본적으로 (정확히 기술적으로 일치하지 않습니다) waiting 리스트와 ready 리스트를 가지고 있다.
그래서 각 작업들의 상태에 따라 해당 리스트에 넣어놓고 관리를 하게 된다.
먼저 ready 리스트에 있는 작업을 꺼내서 하다가, I/O bound 작업을 만나게 된다면 요청을 보낸 뒤에 waiting 리스트에 넣어놓는다.
그리고 waiting 리스트에 있는 작업 중 응답을 받은 작업은 다시 ready 리스트에 저장한다.
이와 같은 형식으로 계속 작업 진행을 하게 되는데, 이것이 파이썬에서 비동기 프로그래밍을 관리하는 원리이다.

비동기 프로그래밍을 우리가 사용하기 위해서는 `asyncio` 라는 파이썬 패키지를 이용해야한다. (이 포스트는 파이썬 3.8 기준이다)
`asyncio` 패키지의 Hello World 예제는 아래와 같다.

```python
import asyncio

async def main():
    print('Hello ...')
    await asyncio.sleep(1)
    print('... World!')

# Python 3.7+
asyncio.run(main())
```

기본적으로는 마지막 줄의 `asyncio.run()` 를 통해 비동기 작업(함수)를 실행하는 구조이다.
여기서 추가적으로 비동기 함수의 정의와 사용을 위해 `async`, `await` 키워드들이 등장하게 되는데 이를 자세히 살펴보자.
먼저, `await`은 비동기 함수의 사용에 대한 '구분선' 역할을 한다.
예컨대, 비동기 작업의 경우 프로세스 입장에서는 "야 A 작업 좀 해와"라고 한 뒤, 아래 있는 다른 코드 B 들을 실행하게 된다.
이 때 아래 있는 코드 B 중 'A 작업'의 결과가 필요한 코드가 있을 수 있다.
따라서 A작업의 결과가 필요한 코드가 실행되기 전에 A 작업의 결과를 기다릴 필요가 있게 된다.
이 때 사용하는 것이 바로 `await` 키워드이다.

수도코드 형식으로 작성해보면 이해가 쉽다.

```c
a = func();
// do something else here
b = await a;
```
`func()` 함수가 비동기 함수라고 가정했을 때, 해당 함수를 불러놓고 다른 일을 하다가(비동기) a가 필요한 시점에 await를 걸어놓음으로써 이 라인에서는 최소 결과값을 받아올 때까지 기다려야한다고 선언한다.
물론 await이 '기다린'다고 해서, `time.sleep()`과 같은 일반적인 형태로 모두 멈춰놓고 기다리는 것은 아니다.
할 수 있는 다른 작업들을 찾아 실행시킬 것이다.
이와 관련해서는 [이 스택오버플로우 글](https://stackoverflow.com/questions/56729764/python-3-7-asyncio-sleep-and-time-sleep#:~:text=When%20time.,await%20statement%20finishes%20its%20execution.)을 살펴보는 것이 좋다.

이렇게 '기다려야할 구분선'을 지어주는 것이 `await` 이라면, `async`는 가장 이해하기 쉽게 `await`를 사용하는 함수에 반드시 붙는 키워드라고 생각하면 된다.
`await`을 쓰려면 `async def`여야 한다라고 외우면 굉장히 편리하다.


## 파이썬 예제 ##

그렇다면 위에서 말했던 것들을 파이썬 예제로 확인해보자.
먼저 아래 코드는 `main()`함수와, `network_bound_job()`이라는 함수로 이루어져 있다.
이 중 `network_bound_job()`은 네트워크를 이용하여 시간이 많이 걸리는 함수라고 가정해보았다.
그 역할은 어떤 url 주소를 넘기면 해당 url의 text를 읽어서 반환하는 것이다.

```python
async def network_bound_job(url):
    print(url)
    await asyncio.sleep(3) # to mimic time-consuming network operation
    return requests.get(url).text[:100]

async def main():
    print(f"started at {time.strftime('%X')}")

    url = "http://naver.com"
    task = asyncio.create_task(network_bound_job(url))
    
    print("While waiting, doing something else here")

    print(await task)

    print("After await task, then do something else here")

    print(f"finished at {time.strftime('%X')}")

asyncio.run(main())
```

`main()`에서 `network_bound_job()`함수를 task로 만들고 실행을 한다.
이 때 사용하는 함수는 `asyncio.create_task()`이다.
그러면 해당 함수는 비동기이기 때문에, await을 보기 전까지 다른 작업을 할 수 있게 된다.
따라서 다른 작업을 하다가, `await task`를 만나면 비동기함수가 끝날 때 까지 기다리게 된다.
이게 끝이다.
실행시간은 3초 (`network_bound_job()`)가 걸리는 것을 확인할 수 있다.

일반적인 상황에서는, network 를 사용하는 비동기 함수를 매우 많이 불러야할 수 있다.
우리의 예제에 비유하자면, 응답을 가져올 url이 많은 경우이다.
그렇다면 아래와 같이 `asyncio.gather()`함수를 사용해주면 된다.
그리고 여기서는 굳이 `asyncio.create_task()`를 사용하지 않아도 된다. (gather에 넘기는 시퀀스가 async def로 선언된 비동기 함수이면 자동으로 task로 전환하기 때문이다)

```python
async def network_bound_job(url):
    print(url)
    await asyncio.sleep(3) # to mimic time-consuming network operation
    return requests.get(url).text[:100]

async def main():
    print(f"started at {time.strftime('%X')}")

    urls = ["http://naver.com", "http://daum.net"]
    group = asyncio.gather(*[network_bound_job(url) for url in urls])
    
    print("While waiting, doing something else here")

    print(await group)

    print("After await task, then do something else here")

    print(f"finished at {time.strftime('%X')}")

asyncio.run(main())
```

이 때 재미있는 점은 아까 말한 것 처럼 `await`을 만났을 때 그냥 무지성으로 기다리는 것이 아니라 다른 작업을 찾아나선다는 것이다.
따라서 `network_bound_job('http://naver.com')`을 실행하던 중 `await asyncio.sleep(3)`을 만나면 바로 뒤이어 `network_bound_job('http://daum.net)`을 실행하게 된다.
결과적으로 실행시간은 동일하게 3초가 걸린 것을 확인할 수 있다.
만약 비동기 프로그래밍을 사용하지 않았다면, 3초 + 3초 == 6초가 되었을 것이다.

## References ##

1. https://realpython.com/async-io-python/
2. https://www.youtube.com/watch?v=oEIoqGd-Sns
3. https://www.youtube.com/watch?v=m0icCqHY39U