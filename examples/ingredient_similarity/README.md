# Ingredient Similarity Search

A standalone tool to find similar ingredients using FlavorGraph embeddings.

## Usage

```bash
# Basic search
python3 run.py --ingredient garlic --topn 10

# Search with specific embeddings file
python3 run.py --ingredient lemon --embeddings ./output/FlavorGraph+CSL-embedding_...pickle

# Search compounds
python3 run.py --ingredient capsaicin --topn 8
```

## How it works

1. **Loads node mapping**: Maps ingredient names to node IDs from the CSV file
2. **Loads embeddings**: Reads the trained embeddings (300-dimensional vectors)  
3. **Fuzzy matching**: Finds ingredients even with partial name matches
4. **Cosine similarity**: Computes similarity in the embedding space
5. **Returns top-N**: Shows most similar ingredients with similarity scores

## Example Output

```
$ python3 run.py --ingredient garlic --topn 5

Loading node mapping: ./input/cleaned/nodes_cleaned_basic.csv
Loading embeddings: ./output/FlavorGraph+CSL-embedding_...pickle

Top similar to: garlic (ID: 2819)
------------------------------------------------------------
 1. campbell's_cheddar_cheese_soup       cosine=0.8183
 2. chicken_rice_pilaf_mix               cosine=0.8084
 3. reduced_fat_chicken_broth            cosine=0.8072
 4. medium_sized_shrimp                  cosine=0.8057
 5. no_salt_added_beef_broth             cosine=0.8047
```

## Features

- **Fuzzy name matching**: Handles partial ingredient names
- **Standalone**: No dependencies on training codebase
- **Fast**: Pure NumPy cosine similarity computation
- **Flexible**: Works with any FlavorGraph embeddings file
