import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from umap import UMAP

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

# 탐색할 파라미터
k_range = range(2, 16)  # k=2~15
umap_dims = [20, 50, 100]  # UMAP 차원
years = [2021, 2022, 2023, 2024, 2025]

# 결과 저장
results = {}

# ============================================================
# 연도별 최적 k, UMAP 차원 탐색
# ============================================================
for year in years:
    year_df = ai_df[ai_df["PBSH"].astype(str).str[:4] == str(year)]

    if len(year_df) < 16:
        print(f"{year}년: 데이터 부족 ({len(year_df)}개)")
        continue

    # 임베딩 추출
    emb_indices = [node_id_map[nid] for nid in year_df["node_id"]]
    p_emb = prob_embs[emb_indices]
    s_emb = sol_embs[emb_indices]

    # L2 정규화
    p_emb = normalize(p_emb, norm='l2')
    s_emb = normalize(s_emb, norm='l2')

    print(f"\n{'='*60}")
    print(f"{year}년: {len(year_df)}개 데이터")
    print(f"{'='*60}")

    for emb_type, emb_data in [("Problem", p_emb), ("Solution", s_emb)]:
        print(f"\n  [{emb_type}]")

        all_scores = []

        for dim in umap_dims:
            # UMAP 차원 축소
            umap_model = UMAP(n_components=dim, min_dist=0.0, metric='cosine', random_state=42)
            emb_reduced = umap_model.fit_transform(emb_data)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
                labels = kmeans.fit_predict(emb_reduced)
                sil_score = silhouette_score(emb_reduced, labels)
                all_scores.append((dim, k, sil_score))

        # 상위 5개 출력
        all_scores_sorted = sorted(all_scores, key=lambda x: x[2], reverse=True)
        print("    Top 5:")
        for rank, (dim, k, score) in enumerate(all_scores_sorted[:5], 1):
            print(f"      {rank}. dim={dim}, k={k}, Silhouette={score:.4f}")

        best_dim, best_k, best_sil = all_scores_sorted[0]

        # 결과 저장
        results[f"{year}_{emb_type}"] = {
            "best_dim": best_dim,
            "best_k": best_k,
            "best_silhouette": best_sil,
            "all_scores": all_scores_sorted[:10]
        }

# ============================================================
# 결과 요약
# ============================================================
print(f"\n{'='*70}")
print("결과 요약 (L2 정규화 + UMAP 차원 축소)")
print(f"{'연도':<6} {'Problem dim':<12} {'Problem k':<10} {'Problem Sil':<12} {'Solution dim':<13} {'Solution k':<11} {'Solution Sil':<12}")


for year in years:
    p_key = f"{year}_Problem"
    s_key = f"{year}_Solution"
    if p_key in results and s_key in results:
        p = results[p_key]
        s = results[s_key]
        print(f"{year:<6} {p['best_dim']:<12} {p['best_k']:<10} {p['best_silhouette']:<12.4f} {s['best_dim']:<13} {s['best_k']:<11} {s['best_silhouette']:<12.4f}")

# optimal_params 형식으로 출력
print(f"\n{'='*70}")
print("clustering.py용 optimal_params:")
print(f"{'='*70}")
print("optimal_params = {")
for year in years:
    p_key = f"{year}_Problem"
    s_key = f"{year}_Solution"
    if p_key in results and s_key in results:
        p = results[p_key]
        s = results[s_key]
        print(f'    {year}: {{"Problem": {{"dim": {p["best_dim"]}, "k": {p["best_k"]}}}, "Solution": {{"dim": {s["best_dim"]}, "k": {s["best_k"]}}}}},')
print("}")

print("\n완료!")
