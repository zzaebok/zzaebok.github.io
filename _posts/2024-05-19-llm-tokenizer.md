---
title: "얼렁뚱땅 LLM을 만들어보자 (1/3)"
date: 2024-05-19 21:21:28 -0400
categories: machine_learning nlp
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>

## Large Language Model (LLM)

2024년 5월 13일, OpenAI에서 GPT-4o를 발표했다.
알파벳 o는 "omni"를 의미하며, GPT가 더 이상 텍스트에 국한되지 않고 오디오, 이미지, 나아가 비디오까지 이해하고 소통할 수 있게 된 것이다.
오냐오냐 하니까 이녀석이 어디까지...

<p align="center">
<img src="https://imgur.com/op3PHGg.png" width="600">
</p>

어쨋든, 더 늦기 전에 LLM을 한 번 학습시켜보고 싶었다.
물론, LLM 자체가 GPT 구조에서 크게 벗어난 것이 아니기 때문에, 특별히 학습에서 다른 점이 있지는 않다.
다만, fine-tuning을 instruction tuning (instruction following) 데이터셋을 통해 한다는 점.
그리고 이 fine-tuning이 모델 전체에 적용되는 것이 아니라, [LoRA](https://arxiv.org/abs/2106.09685)를 이용하여 상대적으로 적은 수의 parameter에만 적용된다는 점이 다를 것이다.

이번 포스트는 토크나이저 학습부터, pre-training, fine-tuning 까지를 다루는 만큼 크게 3개의 포스트로 나눠서 업로드할 예정이다.
또한, 제목에서 시사하는 것처럼 '얼렁뚱땅' 만들 것이기 때문에, 세부적인 내용에서 디테일이 부족할 수 있다.
다만, 전반적으로 LLM이 어떤 방법으로 학습되는 지를 쉽게 파악할 수 있기를 바란다.

## Dataset

먼저 pre-trained 모델 및 tokenizer를 학습시키기 위한 데이터 준비가 필요하다.
LLM, 즉 GPT 스타일의 `Causal` Language Model을 학습하기 위해서는 많은 양의 Text를 이용해야한다.
나는 AIHub의 [`대규모 구매도서 기반 한국어 말뭉치 데이터`](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=653)를 사용할 것이다.
물론 이 뿐만 아니라, [`korean_textbooks`](https://huggingface.co/datasets/maywell/korean_textbooks)과 같은, text 줄글이 아주아주 많은 데이터셋을 사용하면 된다.
데이터는 아래와 같이 생겼다.

<p align="center">
<img src="https://imgur.com/fhJP0uc.png" width="600">
</p>

## Tokenizer 학습

가장 먼저 만들어야하는 것은 Tokenizer이다.
Tokenizer는 Huggingface 의 [tokenizers](https://github.com/huggingface/tokenizers) 라이브러리를 이용하면 쉽게 만들 수 있다.

{% highlight python linenos %}
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, decoders


def main():
    # train tokenizer from scratch
    tokenizer = Tokenizer(
        models.BPE(
            continuing_subword_prefix="##",
        )
    )

    # pre-tokenize
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation(),
        ]
    )

    # model
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        continuing_subword_prefix="##",
    )
    tokenizer.train(["data/pretrain/corpus.txt"], trainer=trainer)

    # decode
    tokenizer.decoder = decoders.WordPiece()

    # wrap and save
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    encoded = wrapped_tokenizer("토크나이저 만들기 성공")
    print(encoded)
    print(wrapped_tokenizer.decode(encoded["input_ids"]))

    wrapped_tokenizer.save_pretrained("model/pretrain")


if __name__ == "__main__":
    main()
{% endhighlight %}


## References
- https://arxiv.org/abs/2106.09685
- https://huggingface.co/learn/nlp-course/en/chapter6/8
