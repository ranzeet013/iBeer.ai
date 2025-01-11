# Beer Info API

This is a simple API built using FastAPI that allows you to retrieve information about different beers, including their history, ingredients, brewing process, and food pairings. The API uses fuzzy search to match beer names and provides detailed information for the best match.

## Features

- Fuzzy search for beer names.
- Fetches detailed information about the beer including:
  - History and background
  - Ingredients
  - Brewing process
  - Notes
  - Food pairing suggestions
  - Smoking preferences
  
## Technologies

- **FastAPI**: Web framework for building APIs.
- **Pydantic**: Data validation and parsing.
- **Fuzzywuzzy**: For performing fuzzy string matching.
- **CORS**: Middleware to handle cross-origin requests.

## Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that you have a JSON file (`data.json`) containing beer information in the following format:

    ```json
    [
      {
        "name_of_beer": "Beer Name",
        "history_and_background": "Some history",
        "ingredients": { "water": "malt", "hops": "yeast" },
        "brewing_process": { "mashing": "description", "boiling": "description", "cooling": "description", "frementation": "description", "conditining": "description", "filtering and packaging": "description" },
        "note": "Some notes",
        "food_pairing": ["food1", "food2"],
        "smokes": ["smoke1", "smoke2"]
      }
    ]
    ```

4. Make sure to replace the file path of the `data.json` in the code with your actual file path.

## Running the API

To run the API locally, execute the following command:

```bash
uvicorn info:app --reload
