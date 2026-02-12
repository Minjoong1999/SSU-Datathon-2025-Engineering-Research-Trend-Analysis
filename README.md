# 🔥 SSU Datathon 2025
팀명: 산소통  
팀장: 박현빈  
팀원: 김다희, 김민중  
<br>
<br>
<br>

## 분석 내용
1. 유사도 기반 도메인 매핑을 통한 **학계간 융합 분석**
2. 융합 연구에서 범용 solution으로 쓰이는 **AI 도메인 심층 분석**  
<br>
<br>
<br>

## 코드 실행 순서
1. preprocess 폴더 실행 (전처리 및 problem & solution 추출)
2. Convergence_Analysis 폴더 실행 (domain간 융합 분석)
3. AI_Domain_Analysis 폴더 실행 (구체적인 porblem과 solution 분석)  
<br>
각 폴더 내의 README.md를 참고해 코드 실행  
<br>
<br>
<br>

## 사용된 주요 Python 라이브러리 요약
### 데이터 처리
pandas - 데이터프레임 조작  
numpy - 수치 연산 및 임베딩 처리  
json - 구조화된 데이터 I/O  
pathlib - 파일 경로 관리  
<br>

### 머신러닝 & 텍스트 처리
scikit-learn (clustering, metrics, preprocessing, feature_extraction)  
BERTopic - 토픽 모델링  
Hugging Face transformers - 임베딩 및 LLM 활용  
PyTorch - 딥러닝 계산  
<br>

### 차원 축소 & 시각화
UMAP - 차원 축소  
matplotlib - 2D/3D 그래프  
seaborn - 통계 시각화 (히트맵 등)  
<br>

### 비동기 & API 통신
asyncio - 비동기 작업 관리  
aiohttp - 비동기 HTTP 요청  
requests - 동기식 HTTP 요청  
google.genai - Google Gemini API  
pdfplumber - PDF 파싱  
<br>

### 유틸리티
tqdm - 진행률 표시  
regex (re) - 정규표현식 (언어 감지)  
collections.Counter - 빈도 계산  
torch - GPU 기반 연산  

<br>
<br>
<br>

## 🚨 주의 사항
로컬 LLM과 임베딩 모델을 사용하므로 24GB 이상의 VRAM을 가진 환경에서 실행을 권장합니다.