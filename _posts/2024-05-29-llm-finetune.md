---
title: 얼렁뚱땅 LLM을 만들어보자 [3/3]
date: 2024-05-28 21:56:28.000000000 -04:00
categories: machine_learning nlp
redirect_to: https://jaebok-lee.com/posts/llm-finetune
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Fine-tuning

이번에는 마지막으로 Fine-tuning을 해볼 차례이다.
Fine-tuning은 Pre-training이 끝난 모델, 즉 어느정도 소양이 있는 모델, 에게 우리가 최종적으로 원하는 Task, 이 포스트의 경우 ChatGPT 같은 문-답, 를 학습시키는 것이다.
다만, 이러한 Fine-tuning은 LLM에 대해서는 Instruction Following 혹은 Instruction Tuning이라는 이름을 가지게 된다.

## Instruction-tuning

그렇다면 왜 "Fine-tuning"이라는 이전부터 많이 쓰였던 단어보다 "Instruction-tuning"이라는 단어를 사용하는 걸까?
사실 Fine-tuning은 모델을 '하나'의 목표 Task에 학습시키는 것을 의미했다.
예컨대 번역, 요약 등이 그 Task의 예시가 된다.
하나의 모델에게 번역을 Fine-tuning 하려면, 한-영 코퍼스를 준비해서 한글 입력이 들어오면 그에 정확히 대응하는 영어를 출력하는 것을 학습시킨다.

<p align="center">
<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fceb4e28d-424d-450d-a29e-598d883d1fb2_1229x500.png" width="700">
</p>

하지만 LLM은 모델 자체가 큰 Capacity를 갖고 있고, 학습된 텍스트 데이터의 양이 어마어마 때문에 여러 가지의 Task를 동시에 수행할 수 있는 능력을 갖게 되었다.
따라서, 이러한 다양한 Task에 대한 input이 들어왔을 때 어떻게 좀 더 정밀하게 output 텍스트를 생성해야하는지 학습할 필요가 생기는 것이다.
Instruction-tuning을 위한 데이터는 이런 다양한 Task 들의 입력과 이에 맞는 모델의 응답으로 구성되어 있다.

특히, 입력의 형태가 굉장히 자연스러운 형태로 변하게 된다.
예를 들어 번역 Task에 대한 기존의 Fine-tuning을 위한 데이터가

> 입력: 나는 저녁을 먹고 싶다.

> 정답: I want to eat dinner.

이었다면, LLM에서의 번역 Task에 대한 Instruction-tuning은 아래와 같이 다양한 형태로 학습될 것이다.

> 입력: 나는 저녁 먹고 싶다는 걸 영어로 어떻게 표현해?

> 정답: You can say "I want to eat dinner."


## Dataset

나는 Instuction-tuning 데이터셋 중 [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a)을 사용했다.
데이터셋은 아래 예시와 같이 형성되어 있다.

<p align="center">
<img src="https://imgur.com/Z1ztnPn.png" width="700">
</p>


## LoRA

자 이제, 모델을 Instruction-tuning (Fine-tuning) 할 차례다.
하지만 문제가 하나 있다.
LLM은 커도 너무 크다는 것이다.
안그래도 Pre-training할 때 시간이랑 돈이 너무 많이 들었는데, 이 큰 모델을 어떻게 또 Fine-tuning 해야할까?
특히, Fine-tuning은 LLM을 운용하고자 하는 기업, 단체에서 원하는 모델이 각각 다르기 때문에 더 큰 문제라고 할 수 있다.

이를 극복하기 위해 우리 훌륭하신 학자 형님들께서 LoRA 라는 방법을 제안하였다.

<p align="center">
<img src="https://imgur.com/qXgbUcZ.png" width="600">
</p>

LoRA는 Low-Rank Adaptation 의 줄임말로, Pre-trained Weight Matrix마다 이에 대응하는 훨씬 낮은 차원의 파라미터를 두고 이 파라미터들만 Fine-tuning 하는 방법이다.
물론, LoRA 자체의 자세한 내용은 이 포스트에서 깊게 다루지는 않을 것이므로 [논문](https://arxiv.org/pdf/2106.09685)을 직접 참고하길 바란다.
아니 뭐 Matrix가 얼마나 차원이 크길래 이런 짓까지 해야하나 싶을 것이다.

공개된 LLM 들의 Hidden Dimension은 각각 OPT (12288), BLOOM (14336), LLaMA 65B (8192) 이다.
즉, 이렇게 매우 큰 차원을 가진 Matrix가 Attention, FFN 등에 마구 마구 사용되고 있다.
따라서 이러한 Matrix를 직접 Fine-tuning하는 것보다 Low-Rank의 파라미터로 치환하여 Fine-tuning 하는 것이 훨씬 학습 속도가 빨라지게 된다.

## 학습 Code

학습은 Huggingface 라이브러리를 이용한다.

{% highlight python linenos %}
import evaluate
import torch
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset
{% endhighlight %}

- Fine-tuning을 위해 크게 두 개의 라이브러리를 추가로 사용한다.
    - `peft`: LoRA를 적용하기 위한 huggingface의 [peft](https://github.com/huggingface/peft) 라이브러리이다. peft는 parameter-efficient fine-tuning 의 줄임말이다.
    - `trl`: Instruction-tuning을 위한 DataCollator를 따로 사용하는데, huggingface의 [trl](https://huggingface.co/docs/trl/en/index) 라이브러리를 이용한다. 사실 직접 구현해도 되는데 귀찮아서 쓰기로 한다.

{% highlight python linenos %}
# load pre-trained model
context_length = 512
model_name = "model/pretrain/multinode"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    additional_special_tokens=["<|prompt|>", "<|assistant|>", "<|endofanswer|>"],
)
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endofanswer|>")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
)
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id
{% endhighlight %}

- Instruction-tuning용 모델에서는 사용자의 input, 모델의 output, 그리고 eos를 표시하기 위한 special token들을 따로 만들어서 tokenizer에 추가해주었다.
- `line 14`: model embedding size를 늘려주는 것도 잊지 않아야한다.
- 이전 Pre-training 때는 pad, eos를 구분하지 않았지만, 이제는 모델 응답을 표현하기 위해 명확한 eos 토큰을 지정해주었다.

{% highlight python linenos %}
# apply LoRA
config = LoraConfig(
    r=256,
    lora_alpha=512,
    lora_dropout=0.0,
    target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"],
)
model = get_peft_model(model, config)
print(model.print_trainable_parameters())
print(model)
{% endhighlight %}

- LoRA를 적용하기 위해서는 먼저 LoraConfig 를 작성해서 `get_peft_model`에 넘겨주어야 한다.
- `line 6`: LoRA를 적용할 module을 선택할 수 있다. 논문에서 Attention을 위한 `q_proj`, `v_proj`에만 LoRA를 적용해도 효과가 좋다는 결과를 보고 동일하게 선택하였다.

{% highlight python linenos %}
# prepare dataset
dataset = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
dataset = dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)

def tokenize(elements):
    texts = []
    for instruction, output in zip(elements["instruction"], elements["output"]):
        text = f"<|prompt|> {instruction} <|assistant|> {output} <|endofanswer|>"
        texts.append(text)

    outputs = tokenizer(
        texts,
        truncation=True,
        max_length=context_length,
    )
    return outputs

tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names,
)
tokenized_dataset = tokenized_dataset.filter(
    lambda example: len(example["input_ids"]) <= context_length
)
print(tokenized_dataset)
{% endhighlight %}

- [Pre-training](https://zzaebok.github.io/machine_learning/nlp/llm-pretrain/)에서처럼 데이터셋을 준비하고 사전에 Tokenize 한다.
- `line 8`: 단, 이번에는 instruction과 기대하는 모델의 output (정답) 을 일정한 형태로 입력으로 넣어주었다. 이 구분을 위한 Special token은 이전에 정의해둔 `<|prompt|>`, `<|assistant|>`, `<|endofanswer|>` 이다.

{% highlight python linenos %}
# prepare evaluation metric
def preprocess_logits_for_metrics(logits, labels):
    ...

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    ...

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer, mlm=False, response_template="<|assistant|>"
)
{% endhighlight %}

- logits의 전처리와 metric은 [Pre-training](https://zzaebok.github.io/machine_learning/nlp/llm-pretrain/)과 동일하게 사용하였다.
- `DataCollator`는 `trl` 라이브러리의 `DataCollatorForCompletionOnlyLM`을 이용하였다. 이는 학습의 입력 문장 중 `response_template`을 포함한 이전 token들의 label을 -100으로 설정하는 것이다. 즉, 모델은 입력을 만드는 것을 학습할 필요 없이, 주어진 입력에 대한 대답 (`response_template` 이후 토큰들) 만을 학습할 것이다.

{% highlight python linenos %}
args = TrainingArguments(
    output_dir="model/finetune",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-6,
    save_steps=100,
    bf16=True,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    # https://discuss.huggingface.co/t/eval-with-trainer-not-running-with-peft-lora-model/53286
    label_names=["labels"],
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

- huggingface의 위대함을 볼 수 있는 부분인 것 같다. LoRA를 integration하는 데에 몇 줄의 코드가 추가되지 않는다. 이전에 보았던 형태의 Trainer 정의와 호출이다.
- 단 현재 버전에서는 문제가 있는지 label_names를 따로 명시해주어야 한다.

## 추론 Code

{% highlight python linenos %}
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model/finetune")
model = AutoPeftModelForCausalLM.from_pretrained("model/finetune").to("cuda")
model.eval()

inputs = tokenizer("<|prompt|> 세상에서 가장 맛있는 음식이 무엇인가요? <|assistant|> ", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items() if k != "token_type_ids"}

outputs = model.generate(
    **inputs,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,
    repetition_penalty=2.0,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])
# <|prompt|> 세상에서 가장 맛있는 음식이 무엇인가요? <|assistant|> 세계에서 제일 맛있다는 음식 중 하나가 바로'랍스터 요리입니다. 하지만, 이 요리는 다른 나라에서는 찾아볼 수 없는 특별한 맛과 식감을 가지고 있습니다! 또한 세계 최고의 레스토랑에서 제공되는 고급스러운 메뉴로 유명합니다만 우리나라에는 없습니다 " <|endofanswer|>
{% endhighlight %}

- peft로 학습된 모델의 추론은 pipeline을 이용하지 않고 직접 tokenize와 generate를 진행하였다.
- `line 8`: 사용자의 입력을 받아 프롬프트를 구성하고 모델에게 input을 전달한다.

## 정리

이렇게, 얼렁뚱땅 LLM을 만들기 위한 세 가지 과정을 모두 거쳤다.
먼저, Tokenizer를 만들었고 대량 코퍼스를 이용해 Pre-training을 하였으며, 문답을 할 수 있도록 이를 Fine-tuning까지 시켜보았다.
그냥 개념으로만 이해하고 있던 내용을 막상 코드로 짜려니까 헷갈리는 부분들이 있었는데 이번 기회에 잘 정리할 수 있어서 좋았다.
huggingface 형님들에게 항상 감사할 따름이다.

## References
- https://arxiv.org/abs/2106.09685
- https://discuss.huggingface.co/t/eval-with-trainer-not-running-with-peft-lora-model/53286
- https://huggingface.co/spaces/PEFT/causal-language-modeling/blob/main/lora_clm_with_additional_tokens.ipynb