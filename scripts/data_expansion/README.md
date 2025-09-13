# Data Expansion Tools

Utilities to expand the FlavorGraph dataset.

## 1) Add Ingredients (with categories)

Seeds CSV format:

```
ingredient,category
black_garlic,Spice
shiso,Plant/Vegetable
fermented_black_bean,Sauce/Powder/Dressing
```

Run:

```bash
python3 scripts/data_expansion/add_ingredients.py \
  --seeds data/expansion/seeds_asian.csv \
  --nodes ./input/cleaned/nodes_cleaned_basic.csv \
  --categories "./input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv"
```

## 2) Add Compounds (with flavor profiles)

Seeds CSV minimum columns:

```
compound,salt,fat,acid,heat,umami,sweet,bitter,aromatic
linalool,0.0,0.1,0.0,0.0,0.0,0.1,0.0,0.8
menthol,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
```

Optional columns: `node_id,original_name,primary_flavor,flavor_strength`

Run:

```bash
python3 scripts/data_expansion/add_compounds.py \
  --seeds data/expansion/compounds_new.csv \
  --nodes ./input/cleaned/nodes_cleaned_basic.csv \
  --mappings ./input/compound_flavors/compound_flavor_mappings.csv
```

## Notes
- New nodes are appended with unique `node_id`s
- Category file gets new rows if ingredient not present
- Compounds default to `node_type=compound` and `is_hub=food`

## 3) Build recipe co-occurrence edges

From the provided recipes dataset (JSON):

```bash
python3 scripts/data_expansion/build_edges_from_recipes.py \
  --recipes ./input/recipes/extracted_recipes.json \
  --output  ./input/edges_from_recipes.csv \
  --scoring normalized
```

Scoring modes:
- simple: 1.0 per pair per recipe
- normalized: 1/(n-1) per pair in an n-ingredient recipe (default)
- weighted: distributes `cooccurrence_score` across pairs
