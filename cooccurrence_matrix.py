"""
Problem-Solution 동시출현행렬 생성
- 각 연도별로 Problem 클러스터와 Solution 클러스터의 동시출현 빈도 계산
- topic_names.txt의 한글 토픽명을 사용하여 라벨링
"""

import pandas as pd
import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
cluster_dir = os.path.join(base_dir, "clustering_results_0201")
topic_names_path = os.path.join(cluster_dir, "topic_naming.txt")
output_dir = os.path.join(base_dir, "cooccurrence_results_0201_copy")
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 짧은 토픽 이름 (히트맵용)
# ============================================================
short_names = {
    2021: {
        "Problem": {
            0: "산업 설비 및 물리 시스템 상태 예측/진단",
            1: "이미지/정보 처리 기술 고도화",
            2: "추천·감성·지식 시스템의 맥락 이해 및 개인화",
            3: "불균형 데이터 환경 인식 최적화",
            4: "제약된 환경의 시간/자원 최적화",
            5: "금융 및 시장 예측",
            6: "악성 행위 탐지 및 사이버 보안"
        },
        "Solution": {
            0: "ANN/수치예측",
            1: "지식그래프/NLP",
            2: "ML/회귀예측",
            3: "퍼지/하이브리드",
            4: "딥러닝/CNN",
            5: "유전알고리즘",
            6: "강화학습",
            7: "얼굴인식/표정",
            8: "GAN/생성AI",
            9: "인지분석",
            10: "CFD/시뮬레이션",
            11: "추천엔진"
        }
    },
    2022: {
        "Problem": {
            0: '품질 불량 검사 및 시각적 탐지',
            1: '데이터 불균형 및 한계 극복',
            2: '물리·에너지 수치해석 및 시뮬레이션',
            3: 'SW 신뢰성 및 보안 검증',
            4: 'UX 기반 맞춤형 추천',
            5: '저전력·하드웨어 최적화',
            6: '재난·질병 조기 경보',
            7: '제조·건설 공정 및 시설 관리',
            8: '주관적 평가의 객관화 및 정량화',
            9: '고위험군 조기 예측 및 맞춤 관리 부재',
            10: '시장 불확실성 및 예측 정확도 한계',
            11: '복잡한 제약 조건 하의 자원 최적화 난제'
        },
        "Solution": {
            0: "CNN/YOLO 탐지",
            1: "ANN/LSTM 예측",
            2: "BERT/지식그래프",
            3: "AI 게임/교육",
            4: "GAN/LSTM 보안",
            5: "강화학습/최적화"
        }
    },
    2023: {
        "Problem": {
            0: '공정 품질 및 상태 예측',
            1: '가변적 환경 하의 대상 인식 및 정보 해석 한계',
            2: '정밀 의료 및 건강 모니터링',
            3: '항공·건축 구조 최적화',
            4: '에너지·금융 변동성 예측',
            5: '악조건 하의 시각적 이상 탐지',
            6: '동작 인식 및 메타버스 응용',
            7: '보안 취약점 및 개인정보 보호',
            8: '보이스피싱 및 가짜 정보 탐지',
            9: '데이터 쏠림 및 편향으로 인한 추천 한계',
            10: '고령화 및 인력 부족 사회 대응',
            11: '교육 성과 및 협업 역량 평가',
            12: '디지털 교육 및 학습자 UX',
            13: '분산 자원 할당 및 운영 최적화',
            14: '개인화된 창작 및 스타일 표현의 한계'
        },
        "Solution": {
            0: "ResNet/YOLO",
            1: "XGBoost/XAI",
            2: "KoBERT/추천",
            3: "ANN/시뮬레이션",
            4: "유전알고리즘"
        }
    },
    2024: {
        "Problem": {
            0: '실시간 연산 및 메모리 최적화의 어려움',
            1: '맞춤형 교육 및 디지털 소외 해소',
            2: '공공 안전 및 위험 자동 식별',
            3: '기후 에너지 및 질환 예후 예측',
            4: '극한 환경 구조물 진단 및 제어',
            5: '데이터 품질 확보 및 불균형 개선',
            6: '비즈니스 데이터 분석 및 권리 보호',
            7: '한국형 언어 모델 및 정책 대응',
            8: '무인 자율 제어 및 전술 최적화',
            9: '감성 및 생체 신호 정밀 분류',
            10: '제조 결함 관리 및 비용 예측',
            11: '게임·문화 콘텐츠 디자인 혁신',
            12: '스마트 농업 및 생산 효율화',
            13: '악조건 하의 정밀 객체 관측',
            14: '맞춤형 상품 추천 및 구매 지원'
        },
        "Solution": {
            0: "LLM/프롬프트",
            1: "앙상블/시계열",
            2: "CNN/어텐션",
            3: "양자강화학습"
        }
    },
    2025: {
        "Problem": {
            0: '불균형 데이터 및 비선형 예측',
            1: '극한 환경 구조물 진단 및 설계',
            2: '건물 에너지 관리 및 최적화',
            3: '예술 디지털화 및 초개인화 창작',
            4: '동적 스케줄링 및 경로 최적화',
            5: '생애주기별 학습 및 건강 관리',
            6: 'UX 최적화 및 윤리적 수용성',
            7: '전문 지식 파편화 및 검증 한계',
            8: '특화 정보 탐색 및 오류 정정',
            9: '실시간 연산 지연 및 자원 부족',
            10: '지능형 보안 위협 및 사기 대응',
            11: '대량 콘텐츠 심사 및 공정성',
            12: '사회적 약자 위험 감지 및 케어',
            13: '한국 사회 법률·안전 현안 대응',
            14: '전장 정보 과부하 및 의사결정'
        },
        "Solution": {
            0: "CFD/ANN 시뮬레이션",
            1: "RAG/LLM",
            2: "에듀테크/생성AI",
            3: "생성AI/3D 비전",
            4: "양자NN/DRL",
            5: "앙상블/DNN 예측",
            6: "유전알고리즘",
            7: "BERT/경량화",
            8: "Attention/3D CNN",
            9: "임베디드/경량화",
            10: "멀티모달/LLM",
            11: "YOLOv8/포즈 추정"
        }
    }
}

# ============================================================
# 1. final_topic_names.txt 파싱
# ============================================================
def parse_topic_names(file_path):
    """
    final_topic_names.txt에서 연도별 Problem/Solution 토픽명 추출
    
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    topic_names = {}
    current_year = None
    current_type = None  # "Problem" or "Solution"

    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 연도 감지 (예: "2021" 또는 "2021 ")
        year_match = re.match(r"^(\d{4})\s*$", line)
        if year_match:
            current_year = int(year_match.group(1))
            topic_names[current_year] = {"Problem": {}, "Solution": {}}
            current_type = None
            continue

        # 타입 감지
        if line.lower() == "problem":
            current_type = "Problem"
            continue
        elif line.lower() == "solution":
            current_type = "Solution"
            continue

        # 헤더 스킵
        if line.startswith("군집 ID"):
            continue

        # 데이터 파싱 (예: "Cluster 0 : 이름," 또는 "{Cluster 0 : 이름,")
        # Clsuter 오타도 처리
        cluster_match = re.match(r"^[{\s]*(?:Cluster|Clsuter)\s*(\d+)\s*:\s*(.+?)[,}]*$", line)
        if cluster_match and current_year and current_type:
            topic_id = int(cluster_match.group(1))
            topic_name = cluster_match.group(2).strip().rstrip(',}')
            topic_names[current_year][current_type][topic_id] = topic_name

    return topic_names

print("토픽 이름 파싱 중...")
topic_names = parse_topic_names(topic_names_path)

# 파싱 결과 확인
for year in sorted(topic_names.keys()):
    print(f"\n{year}년:")
    print(f"  Problem 토픽 수: {len(topic_names[year]['Problem'])}")
    print(f"  Solution 토픽 수: {len(topic_names[year]['Solution'])}")

# ============================================================
# 2. 클러스터링 결과 로드 및 동시출현행렬 생성
# ============================================================
years = [2021, 2022, 2023, 2024, 2025]
all_cooccurrence = {}

for year in years:
    print(f"{year}년 동시출현행렬 생성")

    # 클러스터 결과 로드
    prob_csv = os.path.join(cluster_dir, f"{year}_Problem_clusters.csv")
    sol_csv = os.path.join(cluster_dir, f"{year}_Solution_clusters.csv")

    if not os.path.exists(prob_csv) or not os.path.exists(sol_csv):
        print(f"파일 없음: {year}년 스킵")
        continue

    prob_df = pd.read_csv(prob_csv)
    sol_df = pd.read_csv(sol_csv)

    # node_id 기준으로 병합
    merged = prob_df[["node_id", "cluster"]].merge(
        sol_df[["node_id", "cluster"]],
        on="node_id",
        suffixes=("_prob", "_sol")
    )

    print(f"병합된 문서 수: {len(merged)}")

    # 동시출현행렬 생성 (crosstab)
    cooccurrence = pd.crosstab(
        merged["cluster_prob"],
        merged["cluster_sol"],
        margins=False
    )

    # 모든 토픽이 행렬에 포함되도록 re-index
    prob_topics = sorted(topic_names[year]["Problem"].keys())
    sol_topics = sorted(topic_names[year]["Solution"].keys())

    cooccurrence = cooccurrence.reindex(index=prob_topics, columns=sol_topics, fill_value=0)

    # 토픽 이름으로 인덱스/컬럼 라벨 생성 (짧은 이름 사용)
    prob_labels = [f"P{i}: {short_names[year]['Problem'].get(i, f'Topic {i}')}"
                   for i in prob_topics]
    sol_labels = [f"S{i}: {short_names[year]['Solution'].get(i, f'Topic {i}')}"
                  for i in sol_topics]

    # 라벨이 적용된 행렬 생성
    cooccurrence_labeled = cooccurrence.copy()
    cooccurrence_labeled.index = prob_labels
    cooccurrence_labeled.columns = sol_labels

    all_cooccurrence[year] = {
        "matrix": cooccurrence,  # 숫자 인덱스
        "matrix_labeled": cooccurrence_labeled,  # 라벨 인덱스
        "prob_topics": prob_topics,
        "sol_topics": sol_topics,
        "prob_labels": prob_labels,
        "sol_labels": sol_labels,
        "raw_merged": merged
    }

    print(f"  행렬 크기: {cooccurrence.shape}")

    # CSV 저장 (라벨 포함)
    csv_path = os.path.join(output_dir, f"{year}_cooccurrence_matrix.csv")
    cooccurrence_labeled.to_csv(csv_path, encoding="utf-8-sig")
    print(f"  저장: {csv_path}")

# ============================================================
# 3. 히트맵 시각화 (개별 연도)
# ============================================================
print("\n히트맵 시각화 생성 중...")

for year in years:
    if year not in all_cooccurrence:
        continue

    data = all_cooccurrence[year]
    matrix = data["matrix_labeled"]

    # 행렬 크기에 따라 figure 크기 조정
    n_prob, n_sol = matrix.shape
    fig_width = max(14, n_sol * 1.2)
    fig_height = max(10, n_prob * 0.7)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 히트맵 생성
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#003f5c", as_cmap=True),
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': '동시출현 빈도'},
        annot_kws={"size": 9}
    )

    ax.set_title(f"{year}년 Problem-Solution 동시출현행렬", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Solution 클러스터", fontsize=11)
    ax.set_ylabel("Problem 클러스터", fontsize=11)

    # 라벨 회전 (x축 45도, y축 수평)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    # 저장
    img_path = os.path.join(output_dir, f"{year}_cooccurrence_heatmap.png")
    plt.savefig(img_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  저장: {img_path}")

# ============================================================
# 4. 주요 Problem-Solution 연결 요약
# ============================================================
print("\n" + "="*60)
print("주요 Problem-Solution 연결 요약")
print("="*60)

summary_data = []

for year in years:
    if year not in all_cooccurrence:
        continue

    data = all_cooccurrence[year]
    matrix = data["matrix_labeled"]

    print(f"\n{year}년:")

    # 가장 빈번한 연결 Top 5
    stacked = matrix.stack()
    top_connections = stacked.nlargest(5)

    for (prob_label, sol_label), count in top_connections.items():
        if count > 0:
            print(f"  {prob_label} ↔ {sol_label}: {count}건")
            summary_data.append({
                "year": year,
                "problem": prob_label,
                "solution": sol_label,
                "count": int(count)
            })

# 요약 CSV 저장
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_dir, "top_connections_summary.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"\n요약 저장: {summary_path}")

# ============================================================
# 6. 전체 연도 통합 시각화 (소형 멀티플)
# ============================================================
print("\n전체 연도 통합 시각화 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()

for idx, year in enumerate(years):
    if year not in all_cooccurrence:
        axes[idx].set_visible(False)
        continue

    ax = axes[idx]
    data = all_cooccurrence[year]
    matrix = data["matrix"]

    # 짧은 라벨 사용
    short_prob = [f"P{i}" for i in data["prob_topics"]]
    short_sol = [f"S{i}" for i in data["sol_topics"]]

    matrix_short = matrix.copy()
    matrix_short.index = short_prob
    matrix_short.columns = short_sol

    sns.heatmap(
        matrix_short,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#79ADD2", as_cmap=True),
        linewidths=0.3,
        ax=ax,
        cbar=False,
        annot_kws={"size": 7}
    )

    ax.set_title(f"{year}년", fontsize=12, fontweight='bold')
    ax.set_xlabel("Solution", fontsize=9)
    ax.set_ylabel("Problem", fontsize=9)
    ax.tick_params(labelsize=7)

# 마지막 subplot 숨기기 (5개 연도만 있으므로)
axes[5].set_visible(False)

plt.suptitle("연도별 Problem-Solution 동시출현행렬", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

combined_path = os.path.join(output_dir, "all_years_cooccurrence.png")
plt.savefig(combined_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"저장: {combined_path}")

# ============================================================
# 7. JSON 요약 저장
# ============================================================
json_summary = {}

for year in years:
    if year not in all_cooccurrence:
        continue

    data = all_cooccurrence[year]
    matrix = data["matrix"]

    # 각 Problem별 가장 많이 연결된 Solution
    top_per_problem = []
    for prob_id in data["prob_topics"]:
        if prob_id in matrix.index:
            row = matrix.loc[prob_id]
            top_sol = row.idxmax()
            count = row.max()
            total = row.sum()
            top_per_problem.append({
                "problem_id": prob_id,
                "problem_name": topic_names[year]["Problem"].get(prob_id, ""),
                "top_solution_id": int(top_sol),
                "top_solution_name": topic_names[year]["Solution"].get(top_sol, ""),
                "count": int(count),
                "ratio": round(count / total, 3) if total > 0 else 0
            })

    json_summary[str(year)] = {
        "n_problem_topics": len(data["prob_topics"]),
        "n_solution_topics": len(data["sol_topics"]),
        "total_documents": len(data["raw_merged"]),
        "top_connections_per_problem": top_per_problem
    }

json_path = os.path.join(output_dir, "cooccurrence_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_summary, f, ensure_ascii=False, indent=2)
print(f"JSON 요약 저장: {json_path}")

print("\n" + "="*60)
print("동시출현행렬 생성 완료!")
print(f"결과 저장 위치: {output_dir}")
print("="*60)
