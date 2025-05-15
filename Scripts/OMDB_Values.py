import requests
import json
import os

# Replace 'your_api_key' with your actual OMDB API key
API_KEY = os.environ.get("OMDB_API_KEY")
BASE_URL = 'http://www.omdbapi.com/'

def get_movie_data(title):
    params = {
        't': title,
        'apikey': API_KEY,
        'plot': 'full'  # Get full plot summary
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        if data.get('Response') == 'True':
            return data
        else:
            print(f"Error: {data.get('Error')}")
            return None
            
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def display_movie_info(movie_data):
    if movie_data:
        print(f"Title: {movie_data.get('Title')}")
        print(f"Year: {movie_data.get('Year')}")
        print(f"Director: {movie_data.get('Director')}")
        print(f"Plot: {movie_data.get('Plot')}")
        print(f"IMDb Rating: {movie_data.get('imdbRating')}")
        print(f"Actors: {movie_data.get('Actors')}")
    else:
        print("No movie data to display.")

# Example usage
if __name__ == "__main__":
    movie_title = "Chariots of Fire"
    movie_data = get_movie_data(movie_title)
    display_movie_info(movie_data)