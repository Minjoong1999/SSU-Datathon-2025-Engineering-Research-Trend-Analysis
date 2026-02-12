import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sklearn.preprocessing import normalize

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
jsonl_path = os.path.join(base_dir, "top1.jsonl")
pemb_path = os.path.join(base_dir, "problem.npy")
semb_path = os.path.join(base_dir, "solution_embeddings_with_TTLE.npy")
mapping_path = os.path.join(base_dir, "node_id_mapping.json")

# 데이터 로드
df = pd.read_json(jsonl_path, lines=True)
ai_df = df[(df['mapped_solution_domain'] == "Computational and artificial intelligence") & (df['isSame'] != 1)]

prob_embs = np.load(pemb_path)
sol_embs = np.load(semb_path)

with open(mapping_path, "r") as f:
    node_id_map = json.load(f)

# ============================================================
# 튜닝된 최적 파라미터
# ============================================================
optimal_params = {
    2021: {"Problem": {"dim": 20, "k": 7}, "Solution": {"dim": 100, "k": 12}},
    2022: {"Problem": {"dim": 50, "k": 12}, "Solution": {"dim": 20, "k": 6}},
    2023: {"Problem": {"dim": 100, "k": 15}, "Solution": {"dim": 100, "k": 5}},
    2024: {"Problem": {"dim": 50, "k": 15}, "Solution": {"dim": 20, "k": 4}},
    2025: {"Problem": {"dim": 20, "k": 15}, "Solution": {"dim": 20, "k": 12}},
}

years = [2021, 2022, 2023, 2024, 2025]

# 결과 저장1
all_results = {}

# ============================================================
# 연도별 클러스터링 수행
# ============================================================
for year in years:
    year_df = ai_df[ai_df["PBSH"].astype(str).str[:4] == str(year)].reset_index(drop=True)

    if len(year_df) == 0:
        print(f"{year}년: 데이터 없음")
        continue

    # 임베딩 추출
    emb_indices = [node_id_map[nid] for nid in year_df["node_id"]]
    p_emb = prob_embs[emb_indices]
    s_emb = sol_embs[emb_indices]

    # L2 정규화
    p_emb = normalize(p_emb, norm='l2')
    s_emb = normalize(s_emb, norm='l2')

    # 텍스트 추출
    problem_docs = year_df["problem"].tolist()
    solution_docs = year_df["solution"].tolist()

    print(f"\n{'='*70}")
    print(f"{year}년: {len(year_df)}개 데이터")
    print(f"{'='*70}")

    year_results = {}

    for emb_type, emb_data, docs in [("Problem", p_emb, problem_docs), ("Solution", s_emb, solution_docs)]:
        params = optimal_params[year][emb_type]
        dim = params["dim"]
        k = params["k"]

        print(f"\n  [{emb_type}] dim={dim}, k={k}")

        # UMAP 모델
        umap_model = UMAP(n_components=dim, min_dist=0.0, metric='cosine', random_state=42)

        # KMeans 모델
        cluster_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)

        # 불용어 제거를 위한 Vectorizer
        vectorizer_model = CountVectorizer(stop_words='english')

        # BERTopic 모델
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            verbose=False
        )

        # 클러스터링 수행
        topics, probs = topic_model.fit_transform(docs, embeddings=emb_data)

        # 토픽 정보 추출
        topic_info = topic_model.get_topic_info()
        print(f"    클러스터 수: {len(topic_info)}")

        # 각 토픽별 키워드 출력
        print("    토픽별 키워드:")
        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            if topic_id != -1:  # 노이즈 제외
                keywords = topic_model.get_topic(topic_id)
                keyword_str = ", ".join([word for word, _ in keywords[:5]])
                print(f"      Topic {topic_id} (n={count}): {keyword_str}")

        # 2D UMAP for visualization
        umap_2d = UMAP(n_components=2, min_dist=0.1, metric='cosine', random_state=42)
        emb_2d = umap_2d.fit_transform(emb_data)

        # 결과 저장
        year_results[emb_type] = {
            "topic_model": topic_model,
            "topics": topics,
            "topic_info": topic_info,
            "params": params,
            "emb_2d": emb_2d
        }

    all_results[year] = year_results

# ============================================================
# 결과 저장
# ============================================================
output_dir = os.path.join(base_dir, "clustering_results_0201")
os.makedirs(output_dir, exist_ok=True)

# 연도별 결과 저장
for year in years:
    if year not in all_results:
        continue

    year_df = ai_df[ai_df["PBSH"].astype(str).str[:4] == str(year)].reset_index(drop=True)

    for emb_type in ["Problem", "Solution"]:
        result = all_results[year][emb_type]
        topic_model = result["topic_model"]
        topics = result["topics"]

        # 클러스터 할당 결과 저장 (CSV)
        cluster_df = year_df[["node_id", "problem", "solution", "PBSH"]].copy()
        cluster_df["cluster"] = topics

        # 토픽 키워드 추가
        keyword_map = {}
        for topic_id in set(topics):
            if topic_id != -1:
                keywords = topic_model.get_topic(topic_id)
                keyword_map[topic_id] = ", ".join([word for word, _ in keywords[:5]])
            else:
                keyword_map[topic_id] = "Noise"
        cluster_df["keywords"] = cluster_df["cluster"].map(keyword_map)

        csv_path = os.path.join(output_dir, f"{year}_{emb_type}_clusters.csv")
        cluster_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 토픽 요약 저장 (JSON)
        topic_summary = []
        topic_info = result["topic_info"]
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id != -1:
                keywords = topic_model.get_topic(topic_id)
                topic_summary.append({
                    "topic_id": topic_id,
                    "count": row["Count"],
                    "keywords": [{"word": word, "score": float(score)} for word, score in keywords[:10]]
                })

        json_path = os.path.join(output_dir, f"{year}_{emb_type}_topics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(topic_summary, f, ensure_ascii=False, indent=2)

        print(f"저장: {csv_path}")
        print(f"저장: {json_path}")

# ============================================================
# 결과 요약
# ============================================================
print(f"\n{'='*70}")
print("클러스터링 완료!")
print(f"{'='*70}")

for year in years:
    if year in all_results:
        p_info = all_results[year]["Problem"]["topic_info"]
        s_info = all_results[year]["Solution"]["topic_info"]
        print(f"{year}년 - Problem: {len(p_info)}개 토픽, Solution: {len(s_info)}개 토픽")

print(f"\n결과 저장 위치: {output_dir}")

# ============================================================
# 2D 시각화
# ============================================================
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 5, figsize=(28, 12))

# 색상 팔레트 설정
colors = sns.color_palette("tab20", 20)

for i, year in enumerate(years):
    if year not in all_results:
        continue

    for j, emb_type in enumerate(["Problem", "Solution"]):
        result = all_results[year][emb_type]
        emb_2d = result["emb_2d"]
        topics = np.array(result["topics"])
        k = result["params"]["k"]

        ax = axes[j, i]
        ax.set_facecolor('white')

        unique_topics = sorted(set(topics))

        for topic_id in unique_topics:
            if topic_id == -1:  # 노이즈는 회색으로
                color = '#CCCCCC'
                alpha = 0.4
            else:
                color = colors[topic_id % len(colors)]
                alpha = 0.7

            mask = topics == topic_id
            points = emb_2d[mask].copy()

            # Jitter 추가: 겹치는 점들을 분산시킴
            jitter_strength = 0.3  # 전체 데이터 범위의 약 3%
            jitter = np.random.normal(0, jitter_strength, points.shape)
            points = points + jitter

            # 점 그리기
            ax.scatter(
                points[:, 0], points[:, 1],
                c=[color], alpha=alpha, s=35,
                edgecolors='white', linewidths=0.5,
                label=f'Topic {topic_id}' if topic_id != -1 else 'Noise'
            )

            # Centroid에 토픽 번호 라벨
            if topic_id != -1 and len(points) > 0:
                centroid = points.mean(axis=0)
                ax.annotate(
                    str(topic_id), centroid,
                    fontsize=9, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                             edgecolor=color, linewidth=1.5, alpha=0.9)
                )

        ax.set_title(f'{emb_type} {year} (k={k})', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.tick_params(labelsize=9)

        # 그리드 스타일
        ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout(pad=2.0)
viz_path = os.path.join(output_dir, "clustering_2d_visualization.png")
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"시각화 저장: {viz_path}")

# 투명 배경 버전
viz_path_transparent = os.path.join(output_dir, "clustering_2d_visualization_transparent.png")
plt.savefig(viz_path_transparent, dpi=300, bbox_inches='tight', transparent=True)
print(f"투명 배경 버전 저장: {viz_path_transparent}")

print("완료!")
