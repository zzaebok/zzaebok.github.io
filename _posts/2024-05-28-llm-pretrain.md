---
title: 얼렁뚱땅 LLM을 만들어보자 [2/3]
date: 2024-05-27 21:56:28.000000000 -04:00
categories: machine_learning nlp
redirect_to: https://jaebok-lee.com/posts/llm-pretrain
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Pre-training

이번에는 Pre-training을 해볼 차례이다.
Pre-training은 모델에게 우리가 원하는 Task를 학습시키기 전에 다량의 코퍼스를 먼저 보여줌으로써, 언어에 대해 전반적인 이해를 시키는 과정이다.
다시 말해, 우리는 ChatGPT 같은 대화가 되는 모델을 만들고 싶지만, 그 전에 수많은 코퍼스를 이용해 먼저 기본 소양을 갖추도록 만들어주는 것이다.
왜 이런 과정이 필요한걸까?

답은 간단한데, 크게 아래 두 가지 이유라고 할 수 있다.

1. 언어에 대한 전반적인 소양을 길러준다.
2. 데이터가 부족하다.

1번은 위에서 언급한대로, 많은 텍스트 데이터를 볼 수록 모델이 말을 '그럴 듯'하게 만들어 내거나, 다양한 주제에 대해 기본 지식을 학습하게 된다.
뿐만 아니라, 2번의 이유처럼 실제 우리가 '원하는' 데이터가 많이 없을 수 있다.
예를 들어, 문장 요약을 잘 하는 LLM을 만들고 싶다고 하자.
이 때 (긴 문서, 정답 요약문)의 학습 데이터는 사실 충분하지 않다.
따라서, 추론할 때 학습 데이터에 없는 생소한 주제 (예컨대 카트라이더 드리프트 방법) 가 나온다면 우리 모델은 버벅이며 이상한 대답을 낼 수도 있게 된다.

이를 극복하기 위해, Pre-training 과정에서 엄청나게 많은 일반 데이터를 모델에게 보여주며, 기본적인 지식과 말하는 법을 가르쳐주는 것이다.
우리는 인터넷 세상에서 끝도 없이 풍부한 데이터를 얻을 수 있다.
위키피디아, 나무위키, 각종 커뮤니티의 글, 뉴스 기사 등이다.

## Next Word Prediction

그렇다면, 이 Pre-training에서는 실제로 무엇을 목표로 모델이 학습하게 될까?
어떻게 그냥 인터넷에 널부러져 있는 데이터를 학습에 이용할 수 있을까?

<p align="center">
<img src="https://av-eks-lekhak.s3.amazonaws.com/media/__sized__/article_images/Screenshot_from_2023-07-14_10-54-14-thumbnail_webp-600x300.webp" width="700">
</p>

바로 Next Word Prediction 이라는 Task를 학습에 이용하게 된다.
모델에게 텍스트를 주고 다음에 올 단어를 예측하도록 학습하는 것이다.
이렇게 학습하면 어떤 점이 좋을까?
위의 학습은 정답을 따로 구축하지 않아도 된다. (번역이나 요약 같이)
그저 텍스트 데이터만 있다면, 구멍을 뚫어 놓고 모델 학습을 돌리면 그만이다.

더 중요한 것은 실제로 많은 Task에 대해 간접적으로 학습할 수 있는 기회가 생긴다.
인터넷 세상에 존재하는 수많은 텍스트 중에는 분명히 아래와 같은 텍스트가 존재한다.

>"Hello의 번역은 '안녕'"

이를 Next Word Prediction으로 Pre-training 시킨다고 하면 아래와 같은 형태일텐데,

>"Hello의 번역은 ______"

반대로 말하면, 자연스럽게 모델이 'Hello' 의 번역이 '안녕' 이라는 것을 학습할 수 있게 된다.
요약이나 감정 분석 등의 Task도 모두 이런 방식으로 간접적으로 학습될 수 있다.


## Dataset

이전 [Tokenizer 학습 편](https://zzaebok.github.io/machine_learning/nlp/llm-tokenizer/)에서 Tokenizer 학습에 사용되었던 AIHub의 [`대규모 구매도서 기반 한국어 말뭉치 데이터`](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=653)를 그대로 Pre-training에 사용한다.
데이터는 아래와 같이 생겼다.

<p align="center">
<img src="https://imgur.com/fhJP0uc.png" width="600">
</p>

## Model

모델 학습에 앞서, LLM의 모델 아키텍쳐를 정해야한다.
나는 Microsoft 에서 만든 [Phi-1.5](https://huggingface.co/microsoft/phi-1_5) 모델의 구조를 이용하기로 했다.
많고 많은 LLM 중 이 모델의 구조를 사용한 이유는 딱히 없고 1.3B Parameter의 상대적으로 적은(?) 파라미터 수를 가졌기 때문이다.
물론 다른 LLM 구조를 이용해도 문제가 없다.
다만, Pre-training은 굉장히 오랜 시간이 걸리기 때문에, 본 포스트에서는 작은 모델을 사용했다고 생각해주면 될 것 같다.

## 학습 Code

학습은 Huggingface 라이브러리를 이용한다.

{% highlight python linenos %}
import os
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    PhiForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
{% endhighlight %}

- Pre-training되지 않은 Phi모델을 학습시키기 위해 `PhiForCausalLM`을 직접 import한다.

{% highlight python linenos %}
def load_txt_to_dataset(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    data = {"text": lines}
    dataset = Dataset.from_dict(data)
    return dataset
{% endhighlight %}

- Multi-line으로 이루어진 text 파일을 읽어 huggingface dataset을 만들어주는 코드이다.

{% highlight python linenos %}
# prepare dataset
dataset = load_txt_to_dataset("data/pretrain/corpus.txt")
dataset = dataset.select(range(100_000_000))
dataset = dataset.train_test_split(test_size=0.001, shuffle=True, seed=42)
print(dataset)
{% endhighlight %}

- 전체 Corpus는 3억개가 넘는 문장이 있지만, 이 예제에서는 1억개의 Sentence만 샘플링해서 사용한다. 너무 많아서 학습이 오래걸리기 때문.

{% highlight python linenos %}
# tokenize dataset
context_length = 512
tokenizer = AutoTokenizer.from_pretrained("model/pretrain")

def tokenize(element):
    """
    A text which length is over `context_length` is divided into multiple segments
    """
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    return outputs

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print(tokenized_dataset)
{% endhighlight %}

- `line 2-3`: 최대 입력은 512로 한다. OpenAI 모델들은 이 길이가 4k 또는 8k까지 허용된다. `tokenizer`는 1단계에서 만들었던 [tokenizer](https://zzaebok.github.io/machine_learning/nlp/llm-tokenizer/) 경로를 입력한다.
- `line 5`: 길이가 긴 Text들을 내가 설정한 max_length(512)에 맞게 자르고 tokenize하는 전처리 함수이다. `return_overflow_tokens`는 문장이 매우 길 경우 이를 512 단위의 여러 Segment로 반환하도록 한 것이다. (어차피 우리는 Next Word Prediction을 할 것이기 때문에 하나라도 버리지 않기 위함이다.)

{% highlight python linenos %}
# initialize model
config = AutoConfig.from_pretrained(
    "microsoft/phi-1_5",
    vocab_size=len(tokenizer),
    max_position_embeddings=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = PhiForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Phi-1_5 size: {model_size/1000**3:.1f}B parameters")
{% endhighlight %}

- `line 2`: 기본 Configuration (dimension size, n_heads 등) 은 microsoft의 pre-trained `phi-1_5`모델에서 가져왔다.
- `line 10`: 우리는 Configuration을 이용하여 모델 weights를 새로 생성하기 때문에 `AutoModel`이 아닌 `PhiForCausalLM`을 직접 만든다.

{% highlight python linenos %}
# prepare evaluation metric
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]

    return torch.argmax(logits, axis=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)

    return metric.compute(predictions=preds, references=labels)
{% endhighlight %}

- `line 2`: logits argument 내에 쓸데 없는 tensor들 때문에 CUDA OOM이 나는 것을 방지하기 위한 함수이다. 자세한 내용은 [이 링크](https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13)를 참조하면 된다.
- `line 10`: huggingface `evaluate` 라이브러리에서 `accuracy` metric을 이용해서 evaluation 성능을 중간중간 평가한다. 사실 `accuracy`는 순서 상관 없이 단순히 토큰 등장만을 측정하므로 이 Pre-training에서 올바른 측정 도구가 아니다. 더 정확한 성능 평가를 위해 [다양한 metric](https://huggingface.co/evaluate-metric)을 살펴보아야 한다.
- `line 12`: 모델의 predictions, 그리고 ground-truth labels을 `accuracy` metric에 전달한다. i번째 인덱스의 token에 대해 생성된 prediction[i]은 사실 i+1번째 word에 대한 prediction이기 때문에, 이 1칸 차이를 메꿔준다.

{% highlight python linenos %}
# train
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="model/pretrain/multinode",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=10_000,
    logging_steps=10_000,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_steps=10_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=10_000,
    # fp16=True,
    bf16=True,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
{% endhighlight %}

- huggingface `transformers` 라이브러리의 `Trainer`를 이용하여 학습을 진행한다.
- `line 2`: Text 데이터를 뚝뚝 짤라 집어넣고, Next Word를 학습시키기 때문에 따로 Padding을 사용하지 않아, pad_token을 eos_token으로 사용한다.
- `line 5-25`: Training arguments이며 학습에 쓰이는 hyper parameter 등을 명시해준다.
    - 시간 문제로 training epoch은 2로 설정하였다.
    - precision은 `bf16`을 사용하였다. 일반 부동소수점 `fp16`보다 표현할 수 있는 range가 넓어 Training 과정에서 overflow가 일어나지 않아 학습이 안정적이라고 한다. 자신의 GPU가 `bf16`을 지원하는지 확인해본 뒤 사용해야한다. 필자의 경우 A100을 이용했다.
    - 마지막에 저장할 Best model을 선택할 때는 evaluation accuracy score가 가장 높은 모델을 선택한다.

이렇게 학습을 진행하면 multi-node / multi-gpu 환경에서조차 시간이 꽤 걸리는 것을 확인할 수 있다.
GPU 가격이나 학습에 사용되는 전력량, 데이터 사용량을 생각해봤을 때, 최근 들어 직접 Pre-training을 하는 곳이 현저히 줄어들고 Open-source Pre-trained LLM의 수요가 늘어나는 것을 이해할 수 있었다.

## 추론 Code

{% highlight python linenos %}
import torch
from transformers import pipeline

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="model/pretrain/multinode", device=device, max_new_tokens=32, repetition_penalty=2.0
)
pipe("밥을 먹고 나면 늘 하는 생각이 있는데")[0]['generated_text']
# 밥을 먹고 나면 늘 하는 생각이 있는데, ‘ 아! 이 맛에 사는구나 ’ 이다. 한 끼를 먹어도 맛있는 것을 먹어야 하고 좋은 곳을 가야만 한다며 내 몸을 혹사시킨다 싶은
{% endhighlight %}

- 추론은 `transformers`의 `pipeline`을 이용하면 쉽게 할 수 있다.
- 학습된 모델을 pipeline에 올리고, 입력을 넣어주면 학습이 된대로 Next Word 들에 대해 쭉쭉쭉 주절주절하는 것을 관찰할 수 있다.

## References
- https://huggingface.co/learn/nlp-course/en/chapter7/6
