---
title: ChatGPT를 넘어, LangChain 정리
date: 2023-05-01 13:22:28.000000000 -04:00
categories: machine_learning llm
redirect_to: https://jaebok-lee.com/posts/langchain
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## ChatGPT, LLM의 한계

바야흐로 ChatGPT의 시대이다.
번역, 검색, 코드 작성 등 텍스트를 이용한 대부분의 업무에 사회 전반적으로 영향을 끼치고 있다.
오늘은 이 Large Language Model(LLM) 이 가진 한계와, 이를 극복하기 위한 프롬프트 엔지니어링 방법,
그 중에서도 LangChain이라는 프레임워크에 대한 소개를 하려고 한다.

LLM 모델이 가진 문제점은 대표적으로 '할루시네이션'이 있다.
할루시네이션은 사실이 아닌 내용을 진짜인 것 처럼 떠드는 것을 의미한다.
대표적으로 ChatGPT에게 최신 드라마인 '더 글로리'에 대해 질문하면 아래와 같은 어이없는 대답을 받을 수 있는데,

<img src="https://imgur.com/wOAoTqo.png" width="600">

문동은의 처절한 복수극이 어느새 초특급 버라이어티 쇼가 되어버렸다.
하지만, 이러한 문제점은 LLM을 포함한 모든 머신 러닝 분야의 고질병이다.
현실에서는 매초, 매분, 매시간 새로운 정보들이 생겨나는데, 이를 매번 학습 데이터에 넣고 모델을 학습시킬 수는 없기 때문이다.
특히 학습 시간과 비용이 매우 큰 LLM의 경우 이는 더 큰 문제가 된다.
물론, 이를 극복하기 위해 Continual Learning 같은 기법들이 활발하게 연구되고 있지만, Cascading Forgetting 같은 부작용이 발생하는 것이 현실이다.

최근 들어, 이러한 문제를 프롬프트 엔지니어링으로 해결하려는 시도들이 많이 등장하고 있다.
'에이 그거 가지고 되겠어?'라는 생각이 처음에는 들었지만, 살펴볼 수록 매력적인 방법이라는 것을 알게 되었다.

## Prompt Engineering

프롬프트 엔지니어링은 LLM의 등장과 함께 굉장히 자주 등장하는 단어이다.
이는 바로 ChatGPT 같은 LLM에게 질문을 '잘하자'는 방법론이다.
일각에서는 인공지능이 고도화된 미래에 유망한 직업으로 프롬프트 엔지니어를 꼽기도 한다.

프롬프트 엔지니어링을 이해하기 위해 다시 앞선 '더 글로리' 예제로 돌아가보자.
만약 내가 ChatGPT에 더 글로리에 대한 정보들을 제공해준 뒤 똑같은 질문을 던지게 되면 어떻게 될까?

<img src="https://imgur.com/uOhI2dN.png" width="600">

나는 더 글로리를 위키백과에 검색한 뒤, 오른쪽에 위치한 Box 정보들을 몽땅 긁어다가 입력으로 포함해 주었다.

<img src="https://imgur.com/ut9CaEx.png" width="600">

지금의 예제에서는 사실 저 정보가 구조화 되어있고 작아서 그냥 내가 원하는 답을 얻을 수 있지만, 현실에서는 훨씬 복잡하고 긴 자연어로 되어있을 수도 있다고 상상하자.
나는 단순히 내가 궁금한 것을 위키백과에 쳐서 거기 나온 내용을 복사 붙여넣기 한 것 뿐이다.

<img src="https://imgur.com/ZwGRyVB.png" width="600">

김성주 MC의 버라이어티쇼가 다시, 내가 알던 더 글로리로 바뀌었다.

이처럼, 프롬프트 엔지니어링은 단순히 질문을 더 길게하고 예시를 드는 것을 넘어서는 개념이다.
지금은 위키피디아의 정보를 추가로 입력해주었지만, 구글 검색 결과를 붙여줄 수도 있는 것이고, 특정 데이터베이스의 결과, 나아가 계산기를 이용한 결과 등을 프롬프트에 입력으로 넣어줄 수 있다.
이후 LLM은 내가 전달해준 정보들을 모델의 지식과 잘 결합하여 더 정확한, 더 꼼꼼한 답변을 생성해낼 수 있게 되는 것이다.

## LangChain

나의 질문과 LLM의 답변 사이에 여러 가지 Tool들을 이용해 프롬프트 엔지니어링을 하는 것.
그리고 그러한 과정을 반복해 원하는 답을 찾는 것.
이것이 바로 `LangChain`의 핵심 개념이다.

<img src="https://miro.medium.com/v2/resize:fit:1400/0*BKOvjpzn6SPKs81L.png" width="400">

LLM은 홀로 사용하기에는 불충분하고 여러 가지 소스와 지식을 결합해야 한다는 것이 주요 포인트이다.
아마 Chain이라는 이름이 붙은 것도 한 API(LLM을 포함)의 output이 LLM의 input으로 들어가고, 또 그 output이 다른 LLM의 input으로 들어가는 등 꼬리에 꼬리를 무는 모습을 형상화한 것이지 않을까 생각한다.

`LangChain`에서 프롬프트 엔지니어링으로 사용할 수 있는 [Tool](https://python.langchain.com/en/latest/modules/agents/tools.html)들은 정말 많다.
Google Places API, Google Search API, Wikipedia API 등 프롬프트를 풍부하게 하기 위한 API들이다.
오늘은 그 중에서도 Google Search API를 활용하여, 더 정확한 정보를 반환할 수 있는 Chain을 이용하는 예제에 대해 알아볼 것이다.

## LangChain 사용 준비

먼저 LangChain을 사용하려면 여러 가지 API에 대한 Key를 미리 발급 받아야한다.
LangChain은 패키지 자체로서의 기능이라기 보다는 다른 API들을 잘 결합하여 프롬프트 엔지니어링을 해주는 패키지이기 때문에 각 Tool들에 대한 키 발급은 각 API 서비스로부터 발급받아야 한다.

먼저 [이곳](https://platform.openai.com/)에서 OpenAI 계정을 만들고 [API Key](https://platform.openai.com/account/api-keys)를 발급받는다.
LangChain에서는 OpenAI의 LLM을 이용할 예정이다.

Search API 사용을 위해 `SerpApi` 계정 등록을 마치고 역시 [API Key](https://serpapi.com/manage-api-key)를 발급 받는다.

이후 `pip`를 이용해 라이브러리들을 설치하여 준비를 마친다

```python
$ pip install langchain
$ pip install openai
$ pip install google-search-results
```

## Search Example

이제 `LangChain`을 이용해서 더 글로리 출연진을 물어보겠다.

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
os.environ["SERPAPI_API_KEY"] = "..."
```

우선은 LLM과 Search Tool을 활용하기 위해 발급받은 API들을 환경변수로 등록해야한다.

{% highlight python linenos %}
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("드라마 더 글로리 출연진이 누구야?")
{% endhighlight %}

- `line 6`: OpenAI API를 이용한 LLM 모델을 초기화시켜준다. `temperature`를 0으로 설정함으로써 매번 동일한 대답이 나오도록 설정하였다.
- `line 8`: 질문에 대해 이용할 수 있는 Tool들을 설정한다. 여기서 설정한 Tool을 모두 사용하는 것은 아니며, LLM 이 스스로 어떤 Tool을 사용할 지 정할 수 있다. 사용할 수 있는 Tool들은 [여기](https://python.langchain.com/en/latest/modules/agents/tools.html)에서 확인할 수 있다.
- `line 10`: Agent를 초기화한다. 여기서 Agent는 `LangChain` 내부에서 프롬프트 엔지니어링을 하는 객체라고 생각하면 된다. LLM 에게 전달할 Prompt를 Tool을 이용해 풍부하게 하거나, LLM 의 output을 post-processing하여 다시 LLM의 input으로 넣는 등의 행동을 한다. Agent의 종류는 [여기](https://python.langchain.com/en/latest/modules/agents/agents/agent_types.html)서 확인해볼 수 있다.
- `line 12`: 원하는 질문을 agent에게 던진다.

실행 결과를 보자.

<img src="https://imgur.com/4zEoQip.png" width="600">

더 글로리의 출연진 송혜교, 이도현, 정성일, 김히어라, 차주영을 올바르게 찾아낸 모습이다.

자 이제, Agent의 역할과 Console에 출력된 Chaining output이 나오게 된 과정을 천천히 살펴보자.
Agent는 먼저, 우리의 질문을 LLM 에 던지기 전에 일정한 Format에 맞게 변형한다.
[이 곳](https://github.com/hwchase17/langchain/blob/master/langchain/agents/chat/prompt.py)을 보면 그 형식을 확인할 수 있는데,

```python
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
The only values that should be in the "action" field are: {tool_names}
The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

` ``
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
` ``

ALWAYS use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action:

` ``
$JSON_BLOB
` ``

Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin! Reminder to always use the exact characters `Final Answer` when responding."""
```

이것처럼, 우리 질문을 이런 사전 정의된 포멧 Prompt에 집어넣는 것이다.
또한, 우리가 사전에 정의한 Tool들의 이름도 삽입해서, LLM에게 대답에 필요한 툴을 사용하라고 한다.

조금 풀어쓰자면, 우리는 langchain에게 "드라마 더글로리 출연진이 누구야?"라고 질문했지만, langchain의 agent는 이를 "자 너는 내가 하는 질문에 대답을 잘 해야해. 너는 질문에 따라 계산기랑 인터넷 중 하나를 이용할 수 있어. 이제 시작한다! 드라마 더 글로리 출연진이 누구야?" 로 바꾸는 것이다.

그럼 LangChain Agent는 

```pycon
 I need to find out who the cast of the drama The Glory is.
Action: Search
Action Input: "The Glory cast"
...
```

이런 응답을 만들어내는데, 여기서 `Action` 필드가 Search 인 것을 확인한 뒤 LLM의 output generation을 멈추고, Agent는 `Action Input` 값을 Search API 에 태우게 된다.
그렇게 나온 Search Output은 Agent에 의해 다시 `Observation` 에 삽입된 뒤, 다시 한 번 LLM에게 입력으로 들어간다.
그럼 LLM 입장에서는 이제, Google Search의 결과 또한 활용할 수 있는 데이터 소스가 된 것이다.

결국 Chaining의 과정이 된 것이다.
LLM에게 이용할 Tool을 묻고, Tool을 이용한 결과를 다시 프롬프트에 넣고, 그 output을 또 활용하는 과정이다.
이러한 과정들을 N 번 반복하면서 `Thought` 필드에 정답을 찾았다는 output이 나오는 순간 Agent는 Chaining을 멈추고 정답을 반환하는 것이다.
우리 예제 같은 경우 Search API를 두 번 Chaining 하여 원하는 결과를 얻게 되었다.

## Conclusion

물론 `LangChain`이 완벽한 솔루션이냐고 한다면 당연히 그렇지 않다.
LLM 역시 확률 모델이기 때문에 Format을 마음대로 어겨서 대답하는 경우도 있고, 끝없이 Chaining을 하기도 한다.
뿐만 아니라, Tool의 output이 완전치 못한 경우, 예컨대 Search 결과가 엉망인, 에도 온전한 응답을 기대하기는 어렵다.

어쨋든 이번 기회를 통해 프롬프트 엔지니어링이 단순한 것이 아니라는 것을 알 수 있었다.
또한 이런 엔지니어링 측면의 접근으로 인해, 오히려 LLM의 등장으로 사라질 것 같았던 분야들(검색, NER 등)이 다시금 중요해질 수도 있다는 생각을 하게 되었다.
오랜만에 재미있는 아이디어를 본 것 같아 기분이 좋다.

## References
- https://python.langchain.com/en/latest/index.html#
- https://github.com/hwchase17/langchain
