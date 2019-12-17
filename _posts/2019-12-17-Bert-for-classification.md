---
title: "Bert를 이용한 Classification 예제"
date: 2019-12-17 15:23:28 -0400
categories: Deep_Learning NLP
---

## 서론 ##
시작은 단순하다.
작년 즈음 Bert라는 어마무시한 NLP 모델이 나왔다는 소식을 들었다.
약 11개의 NLP 분야의 SOTA를 갱신하였고, 모델의 크기 자체도 어마어마하다는 말을 듣고 호기심이 증폭되었다.
하지만 1년간 나는 취준과 인턴, 학교 생활 등에 신경을 쓰느라 이를 자세히 볼 겨를이 없었고 이번 기회에 확인해보기로 하였다.

나는 Pytorch를 주로 사용하기 때문에 Pytorch로 된 classification 예제를 열심히 찾았다.
하지만 검색해보았을 때 어떻게 사용할 수 있는 지에 대한 예제가 부족하였다.
또한 여러 가지 상황에 대비한 코드들이 덕지덕지 붙어있어서 단순히 어떻게 돌아가는 지를 확인하고 싶은 나에게는 어지러울 뿐이었다.
따라서, 이번에 완전히 필요한 간단한 코드들만 이용하여 어떻게 Bert를 이용한 Binary classification을 할 수 있는 지 정리하고자 한다.

현재 SOTA를 갱신하고 있는 NLP 모델들은 거의 transformer network를 기반으로 구축되어있다.
그리고 Pytorch로 이를 구현한 라이브러리는 Hugging face의 [pytorch-transformers](https://github.com/huggingface/transformers)이다.

## Transformer and Bert ##
먼저 이 글은 transformer와 bert에 대해 들어본 적이 있고, 대강 개념은 알고 있으며 예제를 찾고 있는 사람을 위해 쓰인 글이다.
따라서 자세한 설명을 덧붙이진 않겠다.
트랜스포머에 대해 시각화가 잘 된 자료는 [이곳](http://jalammar.github.io/illustrated-transformer/)을 참고하면 좋다.
실제 인풋과 아웃풋, 어떻게 langugage 모델이 될 수 있는 지에 대한 [비디오](https://www.youtube.com/watch?v=xhY7m8QVKjo&t=2551s)를 보아도 도움이 될 것이다.

## 예제 ##
본격적으로 '그래서 내용은 이해했는데, 어떻게 쓸 수 있냐고'에 대해 써보겠다.
우선, 나는 [네이버 영화 리뷰 corpus](https://github.com/e9t/nsmc)의 감성 이진분류(긍부정)를 목적으로 예제를 세웠다.
이 예제는 Google Colab에서 GPU를 활용하여 진행하였다.

먼저, pytorch-transformers 라이브러리를 설치하고 필요한 것들을 import 한다.

```python
!pip install pytorch-transformers
```
```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
```
```python
!git clone https://github.com/e9t/nsmc.git
```

그리고, dataframe으로 파일을 읽고 결측치들을 처리해준다.
본 예제에서는 40%만 샘플링하여서 실험하였다.. 너무 많아서 :(

```python
train_df = pd.read_csv('./nsmc/ratings_train.txt', sep='\t')
test_df = pd.read_csv('./nsmc/ratings_test.txt', sep='\t')
```
```python
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df = train_df.sample(frac=0.4, random_state=999)
test_df = test_df.sample(frac=0.4, random_state=999)
```

이후 데이터 셋을 만들어주고,

```python
class NsmcDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label
```
Train에 사용될 DataLoader를 만들어준다.
Batch size는 2로 하였는데, 4부터는 Colab의 CUDA 메모리가 터지기 때문이다..
때문에 굉장히 오래걸리고 NLP가 자원싸움이라는 생각이 많이 들더라.

```python
nsmc_train_dataset = NsmcDataset(train_df)
train_loader = DataLoader(nsmc_train_dataset, batch_size=2, shuffle=True, num_workers=2)
```
이후 과정은 일반적인 LSTM이나 CNN을 사용하는 과정과 동일하다.
Huggingface에서 구현된 Bert는 pytorch의 module클래스를 상속받고 있다.
따라서 이미지 분류 classification task처럼 진행해주면 된다.

from_pretrained('bert-base-multilingual-cased')를 사용함으로써 google에서 [pretrained한 모델](https://github.com/google-research/bert/blob/master/multilingual.md)을 사용할 수 있다.
그리고 bert 소개글에서와 같이 tokenizer는 wordpiece를 만들어 토큰화가 이루어진다.
여기서 포인트는 우리가 구현체 중 'BertForSequenceClassification'모델을 사용하는 것이다.
이 모델은 디폴트로 이진분류가 되어있다.
따라서 multi-label을 하거나 다른 task를 학습하기 위해서는 구현된 다른 서브 모델들을 활용하면 된다.
예컨대, multi-label을 하고 싶으면 기본 BertModel의 classifier layer를 조정해줘야하는데, [이곳](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)을 참고하면 도움이 될 것이다.

```python
device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)
```

트레이닝 과정은 굉장히 심플하다.
optimizer 설정, 일정 주기별로 Loss 찍기 등 흔히 볼 수 있는 코드로 진행이 가능하다.
주의해야하는 것은, 학습 샘플의 인풋이 (batch_size, sequence_length)로 들어간다는 것이다.
따라서 zero-padding을 직접 해줘서 model의 forward에 넣어줘야한다.

```python
optimizer = Adam(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0


model.train()
for epoch in range(epochs):
    
    for text, label in train_loader:
        optimizer.zero_grad()
        
        # encoding and zero padding
        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if itr % p_itr == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr+=1
```
학습과정이 끝나면, testset을 이용하여 Accuracy를 체크해볼 수 있다.

```python
# evaluation
model.eval()

nsmc_eval_dataset = NsmcDataset(test_df)
eval_loader = DataLoader(nsmc_eval_dataset, batch_size=2, shuffle=False, num_workers=2)

total_loss = 0
total_len = 0
total_correct = 0

for text, label in eval_loader:
    encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
    padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
    sample = torch.tensor(padded_list)
    sample, label = sample.to(device), label.to(device)
    labels = torch.tensor(label)
    outputs = model(sample, labels=labels)
    _, logits = outputs

    pred = torch.argmax(F.softmax(logits), dim=1)
    correct = pred.eq(labels)
    total_correct += correct.sum().item()
    total_len += len(labels)

print('Test accuracy: ', total_correct / total_len)
 
```

## 결론 ##
처음에 생각보다 당황했었다.
Bert라는 매우 유명한 모델에 비해 그 예제가 많이 복잡하고 직관적이지 않았다.
나는 여러 기능을 포함하는 거대한 모델을 원한 것이 아니라 단순한 task에 적용함으로써 모델을 이해하고자 했기 때문이다.
아직 학생신분인 나로서는 multi-processing을 이용할 자원도 없었고, 거대한 학습 corpus도 없었다.
그리고 tokenizer를 구성할 시간과 자원도 없었다.
따라서 가장 기본적인 구현체와, 그 구현체를 이용한 이진분류가 어떻게 진행되는 지에 집중해 예제를 작성하였다.
혹시라도 간단한 pytorch bert 예제를 찾으시는 분이 있다면 꼭 도움이 되었으면 좋겠다.
