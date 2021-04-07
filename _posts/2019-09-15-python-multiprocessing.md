---
title: "Python multiprocessing 파이썬 병렬처리"
date: 2019-09-15 13:38:28 -0400
categories: python
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Intro ##
파이썬의 병렬 처리 모듈 multiprocessing에 대해 살펴보도록 하겠습니다.
병렬 처리를 공부하다보면 뭔가 뜨문뜨문 설명이 되어있어서 헷갈릴 때가 많았기 때문에, 제가 실제로 사용할 것 같은 부분들만 추려서 정리해놓으려고 합니다.
Pool, Process와 공유를 위한 Queue, 그리고 Manager에 대해서 살펴보도록 하겠습니다.

## Pool & Process ##
Pool과 Process 모두 병렬 처리를 위해 사용되는데 두 방식에는 차이점이 존재한다.
직관적으로 쉽게 이해하자면, Pool은 말 그대로 처리할 일(inputs)들을 바닥에 뿌려놓고 알아서 분산 처리를 하게 만드는 것이고
Process는 각 프로세스별로 할당량을 명시적으로 정해준 뒤 일을 맡기는 것이다.
추가적으로 [Quora](https://www.quora.com/What-is-the-difference-between-Process-vs-Pool-in-the-multiprocessing-Python-library)에서 확인한 process와 pool의 차이점도 있다.


>‘Process’ halts the process which is currently under execution and at the same time schedules another process. ‘Pool’ on the other hand waits till the current execution in complete and doesn’t schedule another process until the former is complete which in turn takes up more time for execution.
‘Process’ allocates all the tasks in the memory whereas ‘Pool’ allocates the memory to only for the executing process. You would rather end up using Pool when there are relatively less number of tasks to be executed in parallel and each task has to be executed only once.


### Pool 예시 ###
[출처](https://m.blog.naver.com/townpharm/220951524843)
```python
from multiprocessing import Pool
import time
import os
import math

def f(x):
    print("값", x, "에 대한 작업 Pid = ",os.getpid())
    time.sleep(1)
    return x*x

if __name__ == '__main__':
    p = Pool(3)
    startTime = int(time.time())
    print(p.map(f, range(0,10)))  // 함수와 인자값을 맵핑하면서 데이터를 분배한다
    endTime = int(time.time())
    print("총 작업 시간", (endTime - startTime))
```
말그대로 f(0), f(1), ... f(9)를 땅바닥에 뿌려놓고 Pool에 올라가 있는 세 녀석들이 알아서 하나씩 주워 처리하는 방식이다.

### Process 예시 ###
```python
import os

from multiprocessing import Process

def doubler(number):
    // A doubling function that can be used by a process
    
    result = number * 2
    proc = os.getpid()
    print('{0} doubled to {1} by process id: {2}'.format(
        number, result, proc))


if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25]
    procs = []

    for index, number in enumerate(numbers):
        proc = Process(target=doubler, args=(number,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
```
Pool과는 다르게, 하나의 Process가 각자 target함수와 args를 가지고 일을 처리하게 된다.

## Queue & Manager ##
사실 이번 포스트의 중점적인 내용인 이 부분이라고 할 수 있다.
나는 Process나 Pool의 예시처럼 단순히 출력하고 반환하는 것이 아니라 실제로는 그 프로세스의 결과를 리스트나 큐 등에 저장하여 사용할 것이기 때문이다.
따라서 각 프로세스들이 결과를 하나의 리스트나 큐에 저장하는 방법을 알아볼 것이다.
**프로세스 간의 통신은 Pipe로 할 수 있다. 그러나 나는 아직 그 기능이 필요하지 않기 때문에 결과를 sharing 하는 정도에서만 포스팅하겠다 **

### Queue 예시 ###
```python
from multiprocessing import Process, Queue
 
sentinel = -1
 
def creator(data, q):
    // Creates data to be consumed and waits for the consumer
    // to finish processing
    
    print('Creating data and putting it on the queue')
    for item in data:
        q.put(item)
 

def my_consumer(q):
    // Consumes some data and works on it
    // In this case, all it does is double the input
   
    while True:
        data = q.get()
        print('data found to be processed: {}'.format(data))
        processed = data * 2
        print(processed)
 
        if data is sentinel:
            break
 
if __name__ == '__main__':
    q = Queue()
    data = [5, 10, 13, -1]
    process_one = Process(target=creator, args=(data, q))
    process_two = Process(target=my_consumer, args=(q,))
    process_one.start()
    process_two.start()
 
    q.close()
    q.join_thread()
 
    process_one.join()
    process_two.join()
```
컴퓨터과를 다니다보면 흔히 확인할 수 있는 creator - consumer 상황이다. 하나의 queue를 기준으로 하여 creator는 계속 집어넣고 consumer는 계속 꺼내 쓰는 모습이다.


### Manager ###
보통 multiprocessing 모듈을 인터넷에서 찾다보면 Process, Pool, Queue를 설명하는 글이 대부분인데 나는 Manager에 대해서 조금 끄적여보려고한다.
내가 Manager를 찾게 된 것은 Pool과 Queue를 이용하고 싶었는데 그게 잘 되지 않았기 때문이다. Process를 n개 만들어서 하나하나 지정해주는 것보다 뚝뚝 짤라서 바닥에 뿌려놓고
알아서 처리한다음 결과만 합치고 싶었다. 쉽게말해 Manager를 한 명 두고, 얘를 통해서 뭔가 조작한다고 생각하면 편하다. List, Dictionary, Queue 등의 형태로
지원되고 위에서 배운 Queue()보다는 느리다고 하지만 내 생각에 쓰기 편하고 직관적이라 좋은 것 같다. 이 Manager와 [이 포스트](https://beomi.github.io/2017/07/05/HowToMakeWebCrawler-with-Multiprocess/)
를 활용하여 원하는 List에 함수 결과값을 담아오는 코드를 작성하면 다음과 같다.

```python
import multiprocessing
import time
import requests
from bs4 import BeautifulSoup as bs
from multiprocessing import Manager

def get_links():
    req = requests.get('https://naver.com')
    html = req.text
    soup = bs(html, 'html.parser')
    my_titles = soup.select(
        'li > a'
        )
    data = []

    for title in my_titles:
        data.append(title.get('href'))
    return data

def function(result_list, link):
    req = requests.get(link)
    html = req.text
    soup = bs(html, 'html.parser')
    head = soup.select('h1')
    if len(head) != 0:
        result_list.append(head[0].text.replace('\n',''))

if __name__ == "__main__":
    start = time.time()
    pool = multiprocessing.Pool(8)
    m = Manager()
    result_list = m.list()
    all_links = [x for x in get_links() if x.startswith('http')]
    # use starmap to send multiple arguments
    pool.starmap(function, [(result_list, link) for link in all_links])
    pool.close()
    pool.join()
    print("%s"%(time.time()-start))
```
이 코드는 네이버에 들어가 li>a 에 속한 링크를 다 방문하여 h1으로 꾸며진 텍스트를 긁어오는 일을 한다.
주의할 점은 pool에 map되는 함수가 여러가지 arguments를 인자로 받으려면 Pool.map() 대신 Pool.starmap()을 사용해야한다는 점이다.
병렬 처리를 하지 않을 때 23초가 걸리던 것이 병렬처리 후 8초로 줄었다. (약 4배 성능 향상)
multiprocessing.Queue()와 multiprocessing.Manager.Queue()의 차이는 [다음](https://stackoverflow.com/questions/43439194/python-multiprocessing-queue-vs-multiprocessing-manager-queue)에서 확인할 수 있따.

## Conclusion ##
사실 파이토치 데이터 전처리를 하면서 언젠가 multiprocessing을 공부해야겠다 싶었는데 이번에 정리해본다.
포인트는 뿌려놓고 사용하려면 Pool, 일일이 지정시키려면 Process, 공유메모리를 쓰려면 Queue(), 다귀찮고 파이썬방식으로 하고 싶다 하면 Manager()를 사용하는 것이 되겠다.

----

*2020-05-21: python에는 built-in 으로 된 concurrent.futures 라이브러리가 있다. 이 라이브러리는 high-level이기 때문에, 이해하기가 쉽고 사용이 편리하다. 따라서 병렬처리를 하고자한다면 이 라이브러리를 먼저 이용하고 그 다음에 low-level로 위의 방법들을 사용하면 좋을 것이다. 위의 예시들을 비슷하게 사용하려면, ProcessPoolExecutor와 map을 사용하여 받은 result list를 parsing해서 사용하면 될 것 같다.
