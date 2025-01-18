from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()

# Input data model for mentee and mentor data
class RecommendationRequest(BaseModel):
    menti_data: list  # List of dictionaries with mentee data
    mentor_data: list  # List of dictionaries with mentor data

# Utility function to scale similarity scores
def scale_similarity(score, min_target=70, max_target=100):
    """Scale cosine similarity score to the range [min_target, max_target]."""
    return min_target + (max_target - min_target) * score

# 1. JSON 데이터를 받아 처리하는 엔드포인트
@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        # Load mentee and mentor data from the request
        menti_df = pd.DataFrame(request.menti_data)
        mentor_df = pd.DataFrame(request.mentor_data)

        # Combine features into a single string for TF-IDF processing
        menti_df['combined_features'] = menti_df[['introduction', 'major', 'specialty', 'isInterview']].fillna('').apply(
            lambda x: ' '.join(map(str, x)), axis=1
        )
        mentor_df['combined_features'] = mentor_df[['introduction', 'major', 'specialty', 'isInterview', 'personalHistory']].fillna('').apply(
            lambda x: ' '.join(map(str, x)), axis=1
        )

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        menti_tfidf = vectorizer.fit_transform(menti_df['combined_features'])
        mentor_tfidf = vectorizer.transform(mentor_df['combined_features'])

        # Process in batches
        batch_size = 1000
        results = []
        for menti_idx_start in range(0, menti_tfidf.shape[0], batch_size):
            menti_idx_end = min(menti_idx_start + batch_size, menti_tfidf.shape[0])
            menti_batch = menti_tfidf[menti_idx_start:menti_idx_end]
            
            # Calculate similarity for the current batch
            similarity_matrix = cosine_similarity(menti_batch, mentor_tfidf)
            
            # Process each mentee in the batch
            for batch_idx, similarities in enumerate(similarity_matrix):
                menti_idx = menti_idx_start + batch_idx
                best_match_idx = similarities.argmax()
                best_similarity = similarities[best_match_idx]

                # Scale similarity to 70-100 range
                scaled_similarity = scale_similarity(best_similarity)

                # Retrieve mentor details
                mentor_details = mentor_df.loc[best_match_idx, ['name', 'age', 'specialty', 'personalHistory', 'introduction']].to_dict()
                mentor_details['matching_rate'] = f"{scaled_similarity:.2f}%"

                # Add mentee and mentor match details to results
                results.append({
                    "mentee": menti_df.loc[menti_idx, 'name'],
                    "mentor": mentor_details
                })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. CSV 파일에서 데이터를 읽어 처리하는 엔드포인트
@app.get("/recommend-from-csv")
def recommend_from_csv():
    try:
        # Load data from CSV files
        menti_df = pd.read_csv('Menti_Data_Updated.csv')
        mentor_df = pd.read_csv('Mentor_Data_Updated.csv')

        # Combine features into a single string for TF-IDF processing
        menti_df['combined_features'] = menti_df[['introduction', 'major', 'specialty', 'isInterview']].fillna('').apply(
            lambda x: ' '.join(map(str, x)), axis=1
        )
        mentor_df['combined_features'] = mentor_df[['introduction', 'major', 'specialty', 'isInterview', 'personalHistory']].fillna('').apply(
            lambda x: ' '.join(map(str, x)), axis=1
        )

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        menti_tfidf = vectorizer.fit_transform(menti_df['combined_features'])
        mentor_tfidf = vectorizer.transform(mentor_df['combined_features'])

        # Calculate similarity
        similarity_matrix = cosine_similarity(menti_tfidf, mentor_tfidf)

        # Prepare results
        results = []
        for menti_idx, similarities in enumerate(similarity_matrix):
            best_match_idx = similarities.argmax()
            best_similarity = similarities[best_match_idx]
            scaled_similarity = scale_similarity(best_similarity)

            # Retrieve mentor details
            mentor_details = mentor_df.loc[best_match_idx, ['name', 'age', 'specialty', 'personalHistory', 'introduction']].to_dict()
            mentor_details['matching_rate'] = f"{scaled_similarity:.2f}%"

            # Add to results
            results.append({
                "mentee": menti_df.loc[menti_idx, 'name'],
                "mentor": mentor_details
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))