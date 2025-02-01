import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

file_path = '/Users/Raneet/Desktop/untitled folder/recommendationAPI/beerRecommendationData/cleaned_beer_dataset.csv'
dataframe = pd.read_csv(file_path)

app = FastAPI()

dataframe['ingredients'] = dataframe['ingredients'].apply(lambda x: ', '.join(x.split(', ')) if isinstance(x, str) else '')
dataframe['flavor_profile'] = dataframe['flavor_profile'].apply(lambda x: ', '.join(x.split(', ')) if isinstance(x, str) else '')
dataframe['pairings'] = dataframe['pairings'].apply(lambda x: ', '.join(x.split(', ')) if isinstance(x, str) else '')

dataframe['combined_features'] = (
    dataframe['brand'] + " " +
    dataframe['style'] + " " +
    dataframe['ingredients'] + " " +
    dataframe['flavor_profile'] + " " +
    dataframe['pairings'] + " " +
    dataframe['description']
)

vectorizer = CountVectorizer(stop_words='english', binary=True)
feature_matrix = vectorizer.fit_transform(dataframe['combined_features'])
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

def get_beer_info(beer_name):
    if beer_name not in dataframe['beer_name'].values:
        suggested_name = process.extractOne(beer_name, dataframe['beer_name'].unique())[0]
        beer_name = suggested_name

    beer_info = dataframe[dataframe['beer_name'] == beer_name].iloc[0]

    beer_details = {
        'Beer Name': str(beer_info['beer_name']),
        'Brand': str(beer_info['brand']),
        'Style': str(beer_info['style']),
        'ABV': float(beer_info['abv']) if pd.notna(beer_info['abv']) else None,
        'IBU': int(beer_info['ibu']) if pd.notna(beer_info['ibu']) else None,
        'Ingredients': str(beer_info['ingredients']),
        'Flavor Profile': str(beer_info['flavor_profile']),
        'Pairings': str(beer_info['pairings']),
        'Description': str(beer_info['description'])
    }

    return beer_details

@app.get("/recommend/{beer_name}")
def get_recommendations(beer_name: str):
    if beer_name not in dataframe['beer_name'].values:
        suggested_name = process.extractOne(beer_name, dataframe['beer_name'].unique())[0]
        beer_name = suggested_name

    idx = dataframe[dataframe['beer_name'] == beer_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    beer_indices = [i[0] for i in sim_scores if i[0] != idx]
    recommended_beers = dataframe['beer_name'].iloc[beer_indices].drop_duplicates().head(3).tolist()

    recommendations = []
    for beer in recommended_beers:
        beer_details = get_beer_info(beer)
        recommendations.append(beer_details)
    
    return {"Recommended Beers": recommendations}
