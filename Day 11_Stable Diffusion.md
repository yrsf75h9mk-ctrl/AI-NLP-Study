# 생성형 AI와 Stable Diffusion

2026.10.16.    
멋쟁이 사자처럼    
AI 엔지니어 자연어처리(NLP) 과정 6기    

## 1. 금일 학습 요약

이번 학습에서는 U-Net, Autoencoder, VAE, Diffusion, CLIP의 개념을 학습하고, 각 구성 요소가 Stable Diffusion에서 어떻게 연결되는지 살펴보았다. 또한 여러 유형의 데이터를 함께 학습하는 멀티모달 학습과 새로운 콘텐츠를 생성하는 생성형 AI의 차이를 이해했다.

## 2. AI 관련 직무

| 직무 | 주요 역할 |
|---|---|
| AI Researcher | AI 모델과 학습 방법을 연구하고 성능을 개선한다. |
| AI Engineer | AI 모델을 활용해 서비스와 기능을 개발한다. |
| MLOps Engineer | 모델의 배포, 운영, 모니터링 및 학습 파이프라인을 관리한다. |

AI 프로젝트는 모델 개발뿐 아니라 문제 정의, 데이터 처리, 서비스 적용 및 운영까지 연결되는 과정이다.

## 3. U-Net과 이미지 세그멘테이션

U-Net은 인코더와 디코더가 대칭적으로 연결된 신경망 구조로, 이미지 세그멘테이션과 같은 픽셀 단위 예측 작업에 널리 활용된다.

- **Encoder**: 이미지의 공간적 크기를 줄이며 주요 특징을 추출한다.
- **Decoder**: 추출한 특징을 바탕으로 공간적 정보를 복원한다.
- **Skip Connection**: 인코더의 특징을 디코더에 전달해 세부적인 공간 정보를 보존하도록 돕는다.

### 세그멘테이션의 종류

- **Semantic Segmentation**: 각 픽셀을 사람, 자동차, 도로 등의 클래스로 분류한다. 같은 클래스의 개별 객체는 구분하지 않는다.
- **Instance Segmentation**: 같은 클래스에 속한 객체도 개별적으로 구분한다.

U-Net은 원래 세그멘테이션에 널리 사용되는 구조지만, Stable Diffusion에서는 변형된 U-Net이 노이즈 제거에 활용된다.

## 4. Autoencoder와 VAE

### Autoencoder

Autoencoder(오토인코더)는 입력 데이터를 압축된 표현으로 변환한 뒤, 이를 이용해 원래 데이터와 유사한 결과를 복원하도록 학습하는 신경망이다.

- **Encoder**: 입력 데이터를 잠재 표현(Latent Representation)으로 변환한다.
- **Latent Space**: 데이터의 주요 특징을 압축해 표현하는 공간이다.
- **Decoder**: 잠재 표현을 바탕으로 데이터를 복원한다.

### VAE(Variational Autoencoder)

VAE는 잠재 변수를 확률 분포로 모델링하는 생성 모델이다. 인코더가 잠재 분포의 평균과 분산 정보를 추정하고, 그 분포에서 샘플링한 잠재 변수를 디코더에 전달한다.

재매개변수화 트릭(Reparameterization Trick)은 다음과 같이 표현할 수 있다.

$$
z = \mu + \sigma \odot \epsilon
$$

$$
\epsilon \sim \mathcal{N}(0,I)
$$

- \(\mu\): 잠재 분포의 평균
- \(\sigma\): 잠재 분포의 표준편차
- \(\epsilon\): 표준정규분포에서 추출한 무작위 변수
- \(z\): 샘플링된 잠재 변수

무작위 샘플링을 미분 가능한 연산과 분리함으로써 역전파를 통한 학습이 가능하도록 한다.

Stable Diffusion에서는 VAE가 이미지와 잠재 표현 사이의 변환을 담당한다. 이미지를 잠재 공간으로 인코딩하고, 생성이 끝난 잠재 표현을 다시 픽셀 이미지로 디코딩한다.

## 5. Diffusion Model

Diffusion Model(확산 모델)은 데이터에 노이즈를 점진적으로 추가하는 과정과, 노이즈를 제거해 데이터를 복원하는 과정을 학습하는 생성 모델이다.

### Forward Process: 노이즈 추가

원본 데이터에 여러 단계로 노이즈를 추가한다. 일반적인 DDPM의 한 단계는 다음과 같이 표현할 수 있다.

$$
q(x_t \mid x_{t-1})
=
\mathcal{N}
\left(
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
\right)
$$

- \(x_t\): \(t\)단계의 데이터
- \(\beta_t\): 해당 단계의 노이즈 크기를 결정하는 값

### Reverse Process: 노이즈 제거

모델은 노이즈가 포함된 데이터에서 제거해야 할 노이즈를 예측하도록 학습할 수 있다.

$$
\hat{\epsilon} = \epsilon_\theta(x_t,t)
$$

여기서 \(\epsilon_\theta\)는 모델이 예측한 노이즈다. 예측값을 활용해 여러 단계에 걸쳐 데이터를 복원한다. 다만 모든 확산 모델이 노이즈 자체를 예측하는 것은 아니며, 모델에 따라 원본 데이터나 다른 파라미터화된 값을 예측할 수도 있다.

### Latent Diffusion

일반적인 확산 모델은 픽셀 공간에서 직접 노이즈 제거를 수행할 수 있다. 반면 Latent Diffusion Model(잠재 확산 모델)은 압축된 잠재 공간에서 확산 과정을 수행해 계산 부담을 줄인다.

Stable Diffusion은 이러한 잠재 확산 방식을 활용한다. 이미지 전체의 픽셀을 직접 처리하는 대신, VAE가 만든 잠재 표현에서 노이즈 제거를 수행한다.

## 6. 멀티모달 학습과 CLIP

### 멀티모달 학습(Multimodal Learning)

멀티모달 학습은 텍스트, 이미지, 음성, 영상 등 서로 다른 유형의 데이터를 함께 활용해 각 데이터의 특징과 관계를 학습하는 방법이다.

- **모달리티(Modality)**: 정보가 표현되는 방식 또는 데이터 유형
- **멀티모달(Multimodal)**: 두 가지 이상의 모달리티를 다루는 것

예를 들어 이미지와 이미지 설명을 함께 학습하면, 시각적 특징과 텍스트의 의미 사이 관계를 학습할 수 있다.

### CLIP(Contrastive Language–Image Pre-training)

CLIP은 이미지와 텍스트의 관계를 학습하는 대표적인 멀티모달 모델이다.

CLIP은 이미지 인코더와 텍스트 인코더를 사용해 이미지와 문장을 각각 벡터 표현으로 변환한다. 이후 대조 학습(Contrastive Learning)을 통해 관련 있는 이미지와 텍스트의 표현은 가깝게, 관련 없는 쌍의 표현은 멀어지도록 학습한다.

**핵심은 이미지와 텍스트를 각각 처리하는 데 그치지 않고, 두 데이터 사이의 의미적 관계를 학습한다는 것이다.**

### CLIP의 활용: Zero-shot Classification

제로샷 분류는 특정 클래스에 대해 별도로 분류기를 학습하지 않고도, 주어진 후보 텍스트와 이미지의 유사도를 비교해 분류하는 방식이다.

예를 들어 강아지 사진이 주어지면 다음 후보 문장과의 유사도를 비교할 수 있다.

- A photo of a dog
- A photo of a cat
- A photo of a car

CLIP은 이미지와 각 문장의 표현 사이 유사도를 계산해 가장 관련성이 높은 후보를 선택할 수 있다. 다만 결과는 후보 문장과 학습 데이터의 특성 등에 영향을 받는다.

CLIP은 멀티모달 학습의 대표적인 사례이지만, 모든 멀티모달 AI의 유일한 기반은 아니다.

## 7. Stable Diffusion의 구성 요소와 생성 과정

Stable Diffusion은 텍스트 조건을 활용해 새로운 이미지를 생성하는 잠재 확산 모델이다.

### 주요 구성 요소

| 구성 요소 | 역할 |
|---|---|
| Tokenizer | 프롬프트를 토큰으로 변환한다. |
| Text Encoder | 토큰을 텍스트 특징 표현으로 변환한다. |
| U-Net | 노이즈가 포함된 잠재 표현을 바탕으로 노이즈 제거에 필요한 정보를 예측한다. |
| Scheduler | 각 단계의 노이즈 제거 및 잠재 표현 업데이트를 관리한다. |
| VAE Decoder | 최종 잠재 표현을 픽셀 이미지로 변환한다. |

### 이미지 생성 흐름

1. **프롬프트 입력**: 사용자가 원하는 이미지를 텍스트로 설명한다.
2. **텍스트 인코딩**: Tokenizer와 Text Encoder가 프롬프트를 처리한다.
3. **초기 노이즈 생성**: 잠재 공간에서 무작위 노이즈로 시작한다.
4. **반복적 노이즈 제거**: U-Net이 텍스트 조건을 참고해 노이즈를 예측하고, Scheduler가 잠재 표현을 업데이트한다.
5. **이미지 복원**: VAE Decoder가 최종 잠재 표현을 픽셀 이미지로 변환한다.

전체 흐름은 다음과 같다.

`프롬프트 → Tokenizer → Text Encoder → 텍스트 조건`

`초기 잠재 노이즈 → U-Net + 텍스트 조건 → Scheduler를 통한 반복적 노이즈 제거 → VAE Decoder → 최종 이미지`

### CLIP과 Stable Diffusion의 관계

CLIP과 Stable Diffusion은 서로 다른 역할을 담당한다.

- **CLIP**: 이미지와 텍스트의 관계를 학습하고 의미적 유사도를 비교할 수 있다.
- **Stable Diffusion**: 텍스트 조건을 활용해 새로운 이미지를 생성한다.

Stable Diffusion 1.x에서는 CLIP 텍스트 인코더가 프롬프트를 표현하는 데 사용된다. U-Net은 이 텍스트 조건을 활용해 잠재 공간에서 노이즈를 제거하고, VAE Decoder는 결과를 이미지로 복원한다.

모든 Stable Diffusion 버전이 동일한 텍스트 인코더를 사용하는 것은 아니다. 예를 들어 Stable Diffusion 2 계열은 OpenCLIP 기반 텍스트 인코더를 사용하며, SDXL은 두 개의 텍스트 인코더를 활용한다.

### Cross-Attention

Cross-Attention(교차 어텐션)은 한 데이터의 특징이 다른 데이터의 정보를 참고하도록 하는 어텐션 구조다.

Stable Diffusion의 대표적인 U-Net 구조에서는 이미지 잠재 표현의 특징이 텍스트 정보를 참고하도록 Cross-Attention을 활용한다.

- **Query(Q)**: 이미지 잠재 표현에서 어떤 정보를 찾을지 나타낸다.
- **Key(K)**: 텍스트 토큰의 특징을 나타낸다.
- **Value(V)**: 텍스트에서 가져올 정보에 해당한다.

일반적인 어텐션 연산은 다음과 같다.

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

여기서 \(QK^\top\)는 Query와 Key 사이의 유사도를 계산하고, Softmax는 이를 가중치로 변환한다. 이후 가중치에 따라 Value를 결합해 필요한 정보를 반영한다.

대표적인 Stable Diffusion U-Net의 Cross-Attention에서는 이미지 특징에서 Query를 만들고, 텍스트 인코더의 출력에서 Key와 Value를 만든다. 이를 통해 이미지 생성 과정이 프롬프트의 관련 정보를 참고할 수 있다.

## 8. 멀티모달 학습과 생성형 AI의 차이

| 구분 | 멀티모달 학습 | 생성형 AI |
|---|---|---|
| 핵심 관점 | 서로 다른 유형의 데이터 사이 관계를 학습 | 새로운 콘텐츠를 생성 |
| 주요 목적 | 데이터 이해, 의미적 연결 및 정렬 | 텍스트·이미지·음성 등의 생성 |
| 대표 사례 | CLIP | GPT, Stable Diffusion |
| 결과 예시 | 사진과 가장 잘 맞는 문장 찾기 | 프롬프트에 맞는 새로운 이미지 생성 |

멀티모달은 어떤 유형의 데이터를 함께 다루는지에 관한 개념이고, 생성형 AI는 새로운 결과물을 만들어 내는 기능에 관한 개념이다. 따라서 두 개념은 서로 배타적이지 않으며, 하나의 모델이 멀티모달 기능과 생성 기능을 모두 가질 수도 있다.

예를 들어 이미지를 입력받아 내용을 이해하고 텍스트 답변을 생성하는 모델은 멀티모달 입력과 생성 기능을 함께 활용한다.

## 9. 핵심 정리

- **U-Net**: 이미지 특징을 추출하고 복원하는 구조로, Stable Diffusion에서는 노이즈 제거에 필요한 정보를 예측한다.
- **Autoencoder**: 입력 데이터를 잠재 표현으로 압축하고 복원한다.
- **VAE**: 잠재 변수의 확률 분포를 학습하며, Stable Diffusion에서는 이미지와 잠재 표현 사이의 변환을 담당한다.
- **Diffusion Model**: 노이즈를 추가하는 과정과 제거하는 과정을 통해 데이터를 생성한다.
- **Latent Diffusion**: 압축된 잠재 공간에서 확산 과정을 수행한다.
- **CLIP**: 이미지와 텍스트의 관계를 대조 학습으로 학습한다.
- **Cross-Attention**: 이미지 특징이 텍스트 조건의 관련 정보를 참고하도록 한다.
- **Stable Diffusion**: 텍스트 조건, U-Net, Scheduler, VAE 등을 연결해 이미지를 생성한다.
- **멀티모달 학습과 생성형 AI**: 전자는 여러 유형의 데이터 간 관계를 학습하는 데, 후자는 새로운 콘텐츠를 생성하는 데 초점을 둔다.

## 학습 회고

이번 학습을 통해 이미지 생성 모델은 하나의 신경망만으로 동작하는 것이 아니라, 각기 다른 역할을 하는 여러 구성 요소가 연결되어 작동한다는 점을 이해했다.

특히 CLIP은 이미지와 텍스트의 관계를 학습하는 모델이고, Stable Diffusion은 텍스트 조건을 활용해 새로운 이미지를 생성한다는 차이가 중요했다. 또한 U-Net이 노이즈 제거에 필요한 정보를 예측하고, VAE가 잠재 표현을 이미지로 복원한다는 점을 통해 각 구성 요소의 역할을 구분할 수 있었다.

앞으로 AI 모델을 학습하거나 활용할 때는 모델 이름과 기능만 외우기보다, 입력 데이터가 어떤 표현으로 변환되고 각 구성 요소가 어떤 정보를 처리하며 최종 결과가 어떻게 만들어지는지 전체 흐름을 중심으로 이해하고자 한다.

## 참고 자료

1. OpenAI, [CLIP 소개](https://openai.com/index/clip/)
2. Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
3. Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
4. Hugging Face Diffusers, [Stable Diffusion Text-to-Image Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
