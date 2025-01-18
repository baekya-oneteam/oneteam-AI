import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load data from CSV files
menti_file_path = 'Menti_Data_Updated.csv'
mentor_file_path = 'Mentor_Data_Updated.csv'

menti_df = pd.read_csv(menti_file_path)
mentor_df = pd.read_csv(mentor_file_path)

# Randomly sample data for testing
menti_sample = menti_df.sample(n=5, random_state=42)
mentor_sample = mentor_df.sample(n=5, random_state=42)

# Step 1: Graph Construction
G = nx.Graph()

# Add mentee and mentor nodes
for _, row in menti_sample.iterrows():
    G.add_node(row['name'], type='mentee', specialty=row['specialty'], introduction=row['introduction'])

for _, row in mentor_sample.iterrows():
    G.add_node(row['name'], type='mentor', specialty=row['specialty'], introduction=row['introduction'])

# Combine text features and calculate similarities
menti_sample['combined_features'] = menti_sample[['introduction', 'specialty']].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)
mentor_sample['combined_features'] = mentor_sample[['introduction', 'specialty']].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)

vectorizer = TfidfVectorizer()
menti_tfidf = vectorizer.fit_transform(menti_sample['combined_features'])
mentor_tfidf = vectorizer.transform(mentor_sample['combined_features'])

similarity_matrix = cosine_similarity(menti_tfidf, mentor_tfidf)

# Add edges based on similarity scores
threshold = 0.5  # Similarity threshold for adding edges
for i, mentee in enumerate(menti_sample['name']):
    for j, mentor in enumerate(mentor_sample['name']):
        similarity = similarity_matrix[i, j]
        if similarity >= threshold:
            G.add_edge(mentee, mentor, weight=similarity)

# Step 2: Link Prediction using Jaccard Coefficient
jaccard_scores = list(nx.jaccard_coefficient(G))

# Prepare Recommendations
recommendations = []
for u, v, score in sorted(jaccard_scores, key=lambda x: x[2], reverse=True):
    if G.nodes[u]['type'] == 'mentee' and G.nodes[v]['type'] == 'mentor':
        recommendations.append({
            'mentee': u,
            'mentor': v,
            'jaccard_score': score,
            'mentor_details': {
                'specialty': G.nodes[v]['specialty'],
                'introduction': G.nodes[v]['introduction']
            }
        })

# Output recommendations
print("Recommendations:")
for rec in recommendations:
    print(rec)

"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from mangum import Mangum  # AWS Lambda와 FastAPI를 연결

# FastAPI 애플리케이션 생성
app = FastAPI()

# Input data model for mentee and mentor data
class RecommendationRequest(BaseModel):
    menti_data: list  # 멘티 데이터 리스트
    mentor_data: list  # 멘토 데이터 리스트

# Utility function to scale similarity scores
def scale_similarity(score, min_target=70, max_target=100):
    """유사도 점수를 특정 범위 [min_target, max_target]로 변환."""
    return min_target + (max_target - min_target) * score

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    """
    멘티와 멘토 데이터를 기반으로 추천 결과를 반환하는 엔드포인트.
    """
    try:
        # Load mentee and mentor data from the request
        menti_df = pd.DataFrame(request.menti_data)
        mentor_df = pd.DataFrame(request.mentor_data)

        # 멘티와 멘토의 특성을 결합하여 TF-IDF 벡터화
        menti_df['combined_features'] = menti_df[['introduction', 'major', 'specialty', 'isInterview']].fillna('').apply(
            lambda x: ' '.join(map(str, x)), axis=1
        )
        mentor_df['combined_features'] = mentor_df[['introduction', 'major', 'specialty', 'isInterview', 'personalHistory']].fillna('').apply(
            lambda x: ' '.join(map(str, x)), axis=1
        )

        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        menti_tfidf = vectorizer.fit_transform(menti_df['combined_features'])
        mentor_tfidf = vectorizer.transform(mentor_df['combined_features'])

        # 유사도 계산
        similarity_matrix = cosine_similarity(menti_tfidf, mentor_tfidf)

        # 추천 결과 생성
        results = []
        for menti_idx, similarities in enumerate(similarity_matrix):
            best_match_idx = similarities.argmax()
            best_similarity = similarities[best_match_idx]
            scaled_similarity = scale_similarity(best_similarity)

            # 멘토 세부 정보 가져오기
            mentor_details = mentor_df.loc[best_match_idx, ['name', 'age', 'specialty', 'personalHistory', 'introduction']].to_dict()
            mentor_details['matching_rate'] = f"{scaled_similarity:.2f}%"

            # 멘티-멘토 매칭 결과 추가
            results.append({
                "mentee": menti_df.loc[menti_idx, 'name'],
                "mentor": mentor_details
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Lambda 지원 핸들러 추가
handler = Mangum(app)"""