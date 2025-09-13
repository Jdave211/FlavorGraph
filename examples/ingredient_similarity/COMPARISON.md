# Similarity vs. Substitutability: The Key Difference

## The Problem You Identified

You were absolutely right! The original similarity search was showing **co-occurrence patterns** (ingredients that appear together in recipes) rather than **substitutability** (ingredients you can actually swap in recipes).

### Example: Chocolate Search

**❌ Original Similarity Search (Co-occurrence):**
```
Top similar to: chocolate
1. fine_cracker_crumb         cosine=0.8467
2. carnation_instant_milk     cosine=0.8392  
3. roasted_soybean           cosine=0.8363
4. mild_chile                cosine=0.8292
5. lemon_supreme_cake_mix    cosine=0.8288  ← You can't substitute this for chocolate!
```

**✅ New Substitution Search (Category-filtered):**
```
Top substitutes for: chocolate [Bakery/Dessert/Snack]
1. dark_semi_sweet_chocolate  sim=0.8237  ← Actually substitutable!
2. white_chocolate           sim=0.8177  ← Makes sense!
3. instant_chocolate_pudding sim=0.8161  ← Reasonable alternative
4. vanilla_icing            sim=0.8117  ← Could work in some contexts
5. mint_chip                sim=0.8056  ← Flavor variation
```

## The Solution: Category-Aware Substitution

### Key Improvements:

1. **Category Filtering**: Only suggests ingredients from the same functional category
   - `Fruit` → other fruits (apple → pear, pineapple)
   - `Meat/Animal Product` → other proteins (beef → turkey, roast beef)
   - `Bakery/Dessert/Snack` → other dessert ingredients (chocolate → white chocolate)

2. **Functional Equivalence**: Focuses on ingredients that serve similar roles in recipes

3. **Realistic Substitutions**: Results you could actually use in cooking

### Available Categories:
- `Fruit`
- `Plant/Vegetable` 
- `Meat/Animal Product`
- `Bakery/Dessert/Snack`
- `Spice`
- `Nut/Seed`
- `Seafood`
- `Cereal/Crop/Bean`
- `Beverage Alcoholic`
- `Sauce/Powder/Dressing`
- And more...

## Usage Comparison

### For Exploration (Co-occurrence patterns):
```bash
python3 run.py --ingredient garlic --topn 10
# Shows: What commonly appears WITH garlic in recipes
```

### For Cooking (Substitution):
```bash
python3 substitution_search.py --ingredient garlic --topn 10  
# Shows: What you can use INSTEAD OF garlic
```

## Real-World Examples

| Ingredient | Co-occurrence Shows | Substitution Shows |
|------------|-------------------|-------------------|
| **Apple** | Cinnamon, pie crust, oats | Pear, pineapple, other fruits |
| **Garlic** | Cheese soup, chicken broth | Onion, shallot, herbs |
| **Beef** | Potatoes, onions, broth | Turkey, pork, other meats |
| **Chocolate** | Cake mix, milk, nuts | White chocolate, cocoa, vanilla |

The substitution search gives you **actionable culinary advice** rather than just statistical associations!
