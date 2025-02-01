import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = '/content/drive/MyDrive/recommendationBeer/cleaned_beer_dataset.csv'
dataframe = pd.read_csv(file_path)

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
        print(f"Did you mean '{suggested_name}'?")
        beer_name = suggested_name

    beer_info = dataframe[dataframe['beer_name'] == beer_name].iloc[0]

    beer_details = {
        'Beer Name': beer_info['beer_name'],
        'Brand': beer_info['brand'],
        'Style': beer_info['style'],
        'ABV': beer_info['abv'],
        'IBU': beer_info['ibu'],
        'Ingredients': beer_info['ingredients'],
        'Flavor Profile': beer_info['flavor_profile'],
        'Pairings': beer_info['pairings'],
        'Description': beer_info['description']
    }

    return beer_details


def get_recommendations(beer_name, cosine_sim=cosine_sim):
    if beer_name not in dataframe['beer_name'].values:
        suggested_name = process.extractOne(beer_name, dataframe['beer_name'].unique())[0]
        print(f"Did you mean '{suggested_name}'?")
        beer_name = suggested_name

    idx = dataframe[dataframe['beer_name'] == beer_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    beer_indices = [i[0] for i in sim_scores if i[0] != idx]
    recommended_beers = dataframe['beer_name'].iloc[beer_indices].drop_duplicates().head(3).tolist()

    print("Recommended Beers:")
    for beer in recommended_beers:
        print(f"- {beer}")

    for beer in recommended_beers:
        print(f"\nDetails for recommended beer '{beer}':")
        beer_details = get_beer_info(beer)
        for key, value in beer_details.items():
            print(f"{key}: {value}")

input_beer = input("Enter the name of the beer: ").strip()
get_recommendations(input_beer)