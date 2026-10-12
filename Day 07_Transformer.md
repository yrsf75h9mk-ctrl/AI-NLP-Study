# Transformer 구조와 작동 원리

2026.10.12.     
멋쟁이사자처럼    
AI 엔지니어 자연어처리(NLP) 과정 6기       
   

## 1. Transformer의 등장 배경

기존 RNN은 토큰을 순차적으로 처리하기 때문에 병렬 처리에 제약이 있고, 긴 문장에서 멀리 떨어진 단어의 관계를 학습하기 어려울 수 있다.

Transformer는 **Attention을 이용해 위치에 상과 없이 토큰 간 관계를 계산**하며, 학습 시 여러 토큰 위치를 병렬로 처리할 수 있다.

## 2. Transformer 전체 구조

### Encoder

입력 문장의 문맥 정보를 학습한다.

1. Tokenization → Embedding + Positional Encoding
2. Multi-Head Self-Attention
3. Residual Connection + Layer Normalization
4. Feed-Forward Neural Network (FFN)
5. Residual Connection + Layer Normalization
6. 위 과정을 여러 층 반복

최종 출력: (H=[h_0,h_1,\ldots,h_n]\)

각 \(h_i\)는 해당 토큰의 정보와 문맥 정보가 반영된 벡터다.

### Decoder

이전 토큰과 Encoder의 정보를 활용해 다음 토큰을 예측한다.

1. Masked Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Multi-Head Cross-Attention
4. Residual Connection + Layer Normalization
5. FFN
6. Residual Connection + Layer Normalization
7. 위 과정을 여러 층 반복
8. Linear → Logits → Softmax → 다음 토큰 예측

※ 위 구조는 원래 Transformer의 Post-LN 방식 기준이다. 모델에 따라 정규화 방식과 순서는 달라질 수 있다.

## 3. Attention 계산 과정


### Token Embedding: 토큰의 의미와 관련된 정보를 벡터로 표현한다.
### Positional Encoding: 토큰이 문장 내 몇 번째 위치에 있는지에 대한 정보를 제공한다.


### ① Q, K, V 생성

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

* **Query (Q):** 찾고자 하는 정보
* **Key (K):** 관련성을 비교하는 기준
* **Value (V):** 실제로 가져올 정보

Self-Attention은 같은 입력에서 Q, K, V를 만들고, Cross-Attention은 Q는 Decoder에서, K와 V는 Encoder에서 만든다.

### ② Scaled Dot-Product Attention

$$
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

계산 순서:

1. \(QK^T\): 각 Query와 모든 Key의 내적 점수 계산
2. \(\sqrt{d_k}\)로 나누기: 점수 크기 조정
3. Softmax: Attention 가중치 계산
4. 가중치와 V를 곱해 합산: 관련 정보 반영

### ③ Multi-Head Attention

여러 Attention Head에서 병렬로 관계를 계산하고 결과를 결합한다. 다양한 관점의 관계를 학습할 수 있으며 Encoder와 Decoder 모두에서 사용한다.

## 4. Residual Connection, Layer Normalization, FFN

### Residual Connection (잔차 연결)

원래 입력과 변환 결과를 더한다.

$$
y=F(x)+x
$$

기존 정보의 전달과 깊은 신경망의 학습을 돕는다.

### Layer Normalization (층 정규화)

각 토큰의 특징 차원을 기준으로 정규화해 학습을 안정적으로 돕는다.

### Feed-Forward Neural Network (FFN)

각 토큰의 표현을 추가로 가공한다. 보통 선형 변환 → 활성화 함수 → 선형 변환으로 구성된다.

**Encoder 한 층의 핵심 흐름**

Self-Attention → 입력 더하기 + 정규화 → FFN → 입력 더하기 + 정규화

**Decoder 한 층의 핵심 흐름**

Masked Self-Attention → 입력 더하기 + 정규화 → Cross-Attention → 입력 더하기 + 정규화 → FFN → 입력 더하기 + 정규화

## 5. Masked Self-Attention

Decoder는 미래의 정답 토큰을 참고하면 안 되므로 미래 위치의 Attention 점수를 차단한다.

토큰이 3개일 때 사용하는 Causal Mask의 예시:

$$
M=
\begin{bmatrix}
0&-\infty&-\infty\\
0&0&-\infty\\
0&0&0
\end{bmatrix}
$$

점수 행렬에 Mask를 더한 후 Softmax를 적용한다.

$$
A=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)
$$

* \(0\): 참고 가능
* \(-\infty\): 참고 차단
* Softmax 적용 후 차단된 위치의 가중치는 0이 된다.

첫 번째 위치는 첫 번째 토큰만, 두 번째 위치는 첫 번째와 두 번째 토큰만 참고할 수 있다.

## 6. Cross-Attention

Decoder가 Encoder의 입력 정보를 참고하는 과정이다.

$$
Q=ZW_Q,\quad K=HW_K,\quad V=HW_V
$$

* \(Z\): Decoder의 Masked Self-Attention 출력
* \(H=[h_0,\ldots,h_n]\): Encoder 최종 출력

각 Encoder 출력 \(h_i\)는 Key와 Value로 각각 변환된다.

$$
k_i=h_iW_K,\quad v_i=h_iW_V
$$

Decoder의 Query는 모든 Key와 내적해 관련성을 계산하고, Softmax 가중치로 Value를 조합한다.

$$
C=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Self-Attention과 계산 방식은 같지만, Q와 K·V의 출처가 다르다.

## 7. 학습, 추론, 평가

| 구분              | 핵심 내용                                |
| --------------- | ------------------------------------ |
| 학습 (Training)   | 예측과 정답의 차이로 Loss 계산 → 역전파 → 가중치 업데이트 |
| 추론 (Inference)  | 학습된 가중치로 예측 결과 생성                    |
| 평가 (Evaluation) | 예측 결과를 정답과 비교해 성능 측정                 |

* **학습:** 정답 문장의 이전 토큰을 입력으로 사용하는 Teacher Forcing을 활용할 수 있다. Mask를 이용해 미래 토큰을 차단하며 여러 위치를 병렬 계산한다.
* **추론:** 이전에 생성한 토큰을 다음 입력에 추가하며 순차적으로 생성한다.
* **평가:** 평가용 데이터에 추론을 수행한 후 Accuracy, Loss 등 적절한 지표로 성능을 측정한다. 일반적으로 평가 중에는 가중치를 업데이트하지 않는다.

## 핵심 요약

Transformer는 Attention으로 토큰 간 관계를 파악하고, FFN으로 각 토큰의 표현을 가공한다. Residual Connection과 Layer Normalization은 정보 전달과 학습 안정성을 돕는다. Encoder는 입력 문장의 문맥 표현을 만들고, Decoder는 Masked Self-Attention과 Cross-Attention을 통해 이전 토큰과 입력 정보를 활용해 다음 토큰을 예측한다.
