# 머신러닝과 NLP 전처리 이해하기

### 텍스트는 어떻게 AI의 데이터가 되는가?

AI와 자연어처리를 공부하면서 가장 먼저 이해해야 할 부분은 **컴퓨터가 텍스트를 그대로 이해하는 것이 아니라, 데이터를 숫자로 변환하여 처리한다는 것**이었다.

이번 학습에서는 머신러닝과 딥러닝의 기본 개념부터 NLP의 텍스트 전처리, 토큰화, 벡터화까지 전체적인 흐름을 정리했다.

> **Text → Token → Number → Vector → Model**

---

## 1. AI, 머신러닝, 딥러닝의 관계

먼저 AI, 머신러닝, 딥러닝의 관계를 이해할 필요가 있다.

### AI (Artificial Intelligence)

인공지능은 인간의 지능적인 행동을 컴퓨터가 수행하도록 만드는 기술을 의미한다.

예를 들어,

* 음성 인식
* 이미지 인식
* 추천 시스템
* 자연어처리
* 자율주행

등이 AI의 대표적인 활용 사례이다.

### 머신러닝 (Machine Learning)

머신러닝은 AI를 구현하는 방법 중 하나로, 사람이 모든 규칙을 직접 작성하는 대신 **데이터를 통해 패턴과 규칙을 학습하는 방법**이다.

기존의 규칙 기반 시스템에서는 사람이 직접 규칙을 만들어야 했다.

```text
입력 데이터 + 사람이 만든 규칙 → 결과
```

머신러닝에서는 데이터를 이용해 모델의 파라미터를 학습한다.

```text
입력 데이터 + 정답 데이터 → 학습 → 모델
```

### 딥러닝 (Deep Learning)

딥러닝은 머신러닝의 한 분야로, 여러 층의 신경망을 이용하여 데이터의 복잡한 패턴을 학습한다.

```text
AI
└── Machine Learning
    └── Deep Learning
```

---

# 2. 신경망의 기본 구조

딥러닝에서는 **인공신경망(Artificial Neural Network)**&#xC744; 사용한다.

신경망의 기본적인 계산은 입력값에 가중치를 곱하고 편향을 더하는 방식으로 이루어진다.

```text
z = w₁x₁ + w₂x₂ + ... + b
```

* `x` : 입력값
* `w` : 가중치(Weight)
* `b` : 편향(Bias)

계산된 값은 활성화 함수(Activation Function)를 거쳐 다음 층으로 전달된다.

---

## 3. MLP (Multi-Layer Perceptron)

MLP는 여러 개의 퍼셉트론을 층으로 구성한 신경망이다.

기본적으로 다음과 같은 구조를 가진다.

```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

각 층에서는 입력값을 계산하고 다음 층으로 전달한다.

### 주요 구성

* Input Layer: 입력 데이터를 받는 층
* Hidden Layer: 입력 데이터를 변환하고 패턴을 학습하는 층
* Output Layer: 최종 결과를 출력하는 층

---

# 4. Forward Propagation과 Backpropagation

신경망 학습에서는 크게 **순전파와 역전파**가 사용된다.

## Forward Propagation

입력 데이터를 신경망에 넣고 출력값을 계산하는 과정이다.

```text
Input
 ↓
Hidden Layer
 ↓
Output
```

모델이 예측한 값과 실제 정답을 비교하여 **Loss(손실)**&#xB97C; 계산한다.

---

## Backpropagation

손실을 줄이기 위해 각 가중치가 얼마나 수정되어야 하는지 계산하고, 이를 바탕으로 모델의 파라미터를 업데이트한다.

```text
예측
 ↓
Loss 계산
 ↓
Gradient 계산
 ↓
Weight 업데이트
```

즉,

> **Forward Propagation → 예측 → Loss 계산 → Backpropagation → 파라미터 업데이트**

과정을 반복하면서 모델이 학습된다.

---

# 5. 머신러닝의 학습 방식

머신러닝은 학습 데이터의 형태와 목표에 따라 여러 방식으로 나눌 수 있다.

## 지도학습 (Supervised Learning)

정답이 있는 데이터를 사용하여 학습한다.

예:

```text
고객 리뷰 → 긍정
고객 리뷰 → 부정
```

대표적인 문제:

* 분류(Classification)
* 회귀(Regression)

---

## 비지도학습 (Unsupervised Learning)

정답이 없는 데이터에서 패턴이나 구조를 찾는다.

예:

* 고객 군집화
* 문서 군집화
* 데이터의 패턴 탐색

대표적인 방법:

* Clustering
* Dimensionality Reduction

---

## 강화학습 (Reinforcement Learning)

에이전트가 환경과 상호작용하면서 **보상(Reward)**&#xC744; 최대화하는 방향으로 학습한다.

```text
Agent
 ↓
Action
 ↓
Environment
 ↓
Reward
 ↓
Learning
```

---

# 6. NLP에서 전처리가 필요한 이유

자연어처리(NLP)에서는 사람이 사용하는 언어를 컴퓨터가 처리할 수 있는 형태로 변환해야 한다.

컴퓨터는 다음과 같은 문장을 그대로 계산할 수 없다.

```text
I love this movie.
```

따라서 텍스트를 적절하게 정리하고 숫자 형태로 변환해야 한다.

전체적인 과정은 다음과 같다.

```text
Raw Text
   ↓
Cleaning
   ↓
Normalization
   ↓
Tokenization
   ↓
Numericalization
   ↓
Vectorization
   ↓
Model
```

---

# 7. 텍스트 전처리

텍스트 전처리는 모델이 텍스트를 효과적으로 처리할 수 있도록 데이터를 정리하는 과정이다.

대표적인 전처리 과정은 다음과 같다.

### Cleaning

불필요한 문자나 데이터를 제거한다.

예:

```text
Hello!!! :)
```

→

```text
Hello
```

### Normalization

표현을 일정한 형태로 통일한다.

예:

```text
Hello
hello
HELLO
```

필요한 경우 동일한 형태로 정규화할 수 있다.

### Length Adjustment

문장마다 길이가 다르기 때문에 모델의 입력 형태에 맞게 길이를 조정할 수 있다.

* Padding
* Truncation

---

# 8. Tokenization

토큰화(Tokenization)는 텍스트를 **처리 가능한 작은 단위인 Token으로 나누는 과정**이다.

예를 들어 문장을 단어 단위로 나누면:

```text
I love natural language processing
```

↓

```text
[I, love, natural, language, processing]
```

이 각각의 단어가 Token이 된다.

토큰화 방법에 따라 단어, 문자, Subword 등 다양한 단위를 사용할 수 있다.

---

# 9. Vocabulary와 Token ID

토큰화한 단어를 모델이 처리하기 위해서는 각 토큰에 숫자 ID를 부여할 수 있다.

예를 들어:

```text
hello → 1
world → 2
AI → 3
data → 4
```

이처럼 토큰과 숫자 ID의 대응 관계를 저장한 것이 **Vocabulary**이다.

결과적으로:

```text
"hello world"
```

↓

```text
[1, 2]
```

와 같이 변환할 수 있다.

---

# 10. OOV (Out Of Vocabulary)

Vocabulary에 존재하지 않는 단어가 입력되는 경우가 있다.

이를 **OOV(Out Of Vocabulary)**&#xB77C;고 한다.

예:

```text
Vocabulary:
hello
world
AI
```

새로운 단어:

```text
ChatGPT
```

이 Vocabulary에 없다면 해당 단어를 처리하기 어려워진다.

이러한 문제를 해결하기 위한 방법 중 하나가 **Subword Tokenization**이다.

---

# 11. Subword Tokenization

Subword Tokenization은 단어 전체를 하나의 토큰으로 처리하는 대신 **단어를 더 작은 단위로 나누어 처리하는 방법**이다.

대표적인 방법:

* BPE (Byte Pair Encoding)
* WordPiece
* Unigram

예를 들어 새로운 단어가 등장하더라도 이미 학습된 Subword 단위를 조합하여 처리할 수 있다.

이 방식은 OOV 문제를 줄이는 데 도움이 된다.

---

# 12. 텍스트를 숫자로 표현하기

텍스트를 모델에 입력하기 위해서는 숫자 벡터로 변환해야 한다.

대표적인 방법은 다음과 같다.

```text
BOW
TF-IDF
Word2Vec
Embedding
```

---

## BOW (Bag of Words)

문서에서 단어가 등장했는지를 기반으로 텍스트를 표현하는 방법이다.

예:

```text
I love AI
I love data
```

Vocabulary:

```text
[I, love, AI, data]
```

문서를 숫자로 표현하면 단어의 등장 횟수 등을 이용하여 벡터로 나타낼 수 있다.

장점:

* 단순하고 이해하기 쉽다.
* 구현이 쉽다.

단점:

* 단어의 순서나 문맥을 충분히 반영하기 어렵다.

---

# 13. TF-IDF

TF-IDF는 문서 내에서 특정 단어가 얼마나 중요한지를 계산하는 방법이다.

### TF (Term Frequency)

특정 단어가 한 문서에서 얼마나 자주 등장하는지를 나타낸다.

### IDF (Inverse Document Frequency)

여러 문서에서 너무 흔하게 등장하는 단어의 중요도를 낮춘다.

즉,

> 특정 문서에서는 자주 등장하지만 전체 문서에서는 흔하지 않은 단어에 높은 가중치를 부여한다.

TF-IDF는 문서 분류, 검색 등의 문제에 활용할 수 있다.

---

# 14. Word2Vec

Word2Vec은 단어를 벡터로 표현하는 대표적인 방법이다.

핵심 아이디어는 **비슷한 문맥에서 등장하는 단어는 비슷한 의미를 가진다**는 것이다.

예를 들어:

```text
king
queen
man
woman
```

등의 단어를 학습하면서 단어 간의 관계를 벡터 공간에 표현할 수 있다.

기존 BOW 방식과 달리 단어의 의미와 관계를 벡터로 표현할 수 있다는 특징이 있다.

---

# 15. Embedding

Embedding은 텍스트를 **밀집된(Dense) 벡터 형태로 표현하는 방법**이다.

예:

```text
"AI"
```

↓

```text
[0.21, -0.14, 0.73, ...]
```

이러한 벡터 표현을 통해 모델은 단어 사이의 의미적 관계를 학습할 수 있다.

NLP에서 Embedding은 매우 중요한 개념이며, 이후 Transformer와 LLM을 이해하는 데에도 연결된다.

---

# 16. NLP 전체 흐름 정리

지금까지 배운 내용을 하나의 흐름으로 정리하면 다음과 같다.

```text
문장
 ↓
Text Cleaning
 ↓
Normalization
 ↓
Tokenization
 ↓
Token ID
 ↓
Embedding / Vectorization
 ↓
Model
 ↓
Prediction
```

결국 NLP의 핵심은 다음과 같이 정리할 수 있다.

> **자연어를 컴퓨터가 계산할 수 있는 숫자 표현으로 변환하고, 이를 모델이 학습할 수 있도록 만드는 것**

---

# 17. 이번 학습에서 정리한 핵심 개념

### AI

인간의 지능적인 작업을 컴퓨터가 수행하도록 하는 기술

### Machine Learning

데이터에서 패턴을 학습하여 예측이나 판단을 수행하는 방법

### Deep Learning

다층 신경망을 이용하여 복잡한 패턴을 학습하는 방법

### Tokenization

텍스트를 처리 가능한 단위로 나누는 과정

### Vocabulary

토큰과 ID의 대응 관계를 저장한 집합

### OOV

Vocabulary에 존재하지 않는 단어

### Subword Tokenization

단어를 더 작은 단위로 나누어 처리하는 방법

### Vectorization

텍스트를 숫자 벡터로 변환하는 과정

### Embedding

텍스트의 의미와 관계를 벡터 공간에 표현하는 방법

---

# 18. 앞으로의 학습

이번 학습을 통해 NLP 모델이 텍스트를 처리하는 기본적인 흐름을 이해했다.

앞으로는 다음과 같은 순서로 학습을 확장해보고자 한다.

```text
RNN
 ↓
LSTM
 ↓
Attention
 ↓
Transformer
 ↓
BERT
 ↓
GPT
 ↓
LLM
 ↓
AI Agent
```

특히 Transformer의 Attention 구조가 기존의 RNN 계열 모델과 어떤 차이가 있는지, 그리고 BERT와 GPT가 Transformer를 어떻게 활용하는지 더 자세히 학습할 예정이다.

---

# 19. Business 관점에서의 활용

NLP를 공부하면서 기술 자체뿐만 아니라 **비즈니스 문제에 어떻게 활용할 수 있는지**도 함께 생각해보고 있다.

예를 들어 고객 VOC를 분석한다고 가정하면:

```text
고객 VOC
 ↓
텍스트 전처리
 ↓
Tokenization
 ↓
Embedding
 ↓
감성 분석 / 분류
 ↓
고객 불만 유형 파악
 ↓
비즈니스 개선
```

단순히 텍스트를 분석하는 것에서 끝나는 것이 아니라,

**데이터 → 분석 → 인사이트 → 의사결정**

으로 연결하는 것이 AI를 비즈니스에 활용하는 중요한 과정이라고 생각한다.

---

## 마무리

이번 학습을 통해 AI가 자연어를 처리하는 기본적인 흐름을 정리할 수 있었다.

처음에는 텍스트를 그대로 AI 모델에 입력하면 될 것이라고 생각했지만, 실제로는

> **Text → Token → Number → Vector → Model**

이라는 여러 단계를 거쳐야 한다.

앞으로는 이러한 기초 개념을 바탕으로 실제 데이터를 활용한 NLP 프로젝트를 진행하면서 **AI와 데이터 분석을 실제 비즈니스 문제 해결에 연결하는 방법**을 학습해보고자 한다.
