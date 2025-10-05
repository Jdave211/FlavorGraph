"""Quick verification script for recipe_structures.py"""

from recipe_structures import (
    Recipe, Ingredient, RecipeTemplate, IngredientRole,
    CUISINE_TEMPLATES, COOKING_TECHNIQUES,
    infer_ingredient_role, get_cuisine_template, list_available_cuisines
)

# Test 1: Verify dataclasses exist and can be instantiated
print("Test 1: Creating dataclass instances...")
ingredient = Ingredient(
    name="chicken",
    quantity=1.5,
    unit="lbs",
    role=IngredientRole.PROTEIN.value
)
print(f"✓ Ingredient created: {ingredient}")

recipe = Recipe(
    title="Test Recipe",
    cuisine="Italian",
    servings=4,
    prep_time=15,
    cook_time=30,
    ingredients=[ingredient],
    instructions=["Step 1", "Step 2"]
)
print(f"✓ Recipe created: {recipe.title}")

# Test 2: Verify CUISINE_TEMPLATES has 10+ cuisines
print(f"\nTest 2: Checking cuisine templates...")
print(f"Number of cuisine templates: {len(CUISINE_TEMPLATES)}")
assert len(CUISINE_TEMPLATES) >= 10, "Should have at least 10 cuisine types"
print(f"✓ Has {len(CUISINE_TEMPLATES)} cuisine templates (requirement: 10+)")

# Test 3: List all cuisines
print(f"\nTest 3: Available cuisines:")
cuisines = list_available_cuisines()
for cuisine in cuisines:
    print(f"  - {cuisine}")
print(f"✓ All cuisines listed")

# Test 4: Verify COOKING_TECHNIQUES list exists
print(f"\nTest 4: Checking cooking techniques...")
print(f"Number of cooking techniques: {len(COOKING_TECHNIQUES)}")
print(f"Sample techniques: {COOKING_TECHNIQUES[:5]}")
assert len(COOKING_TECHNIQUES) > 0, "Should have cooking techniques"
print(f"✓ Has {len(COOKING_TECHNIQUES)} cooking techniques")

# Test 5: Verify ingredient role categories
print(f"\nTest 5: Checking ingredient roles...")
roles = [role.value for role in IngredientRole]
print(f"Available roles: {roles}")
required_roles = ['protein', 'aromatic', 'acid', 'fat', 'starch', 'vegetable', 'herb', 'spice']
for role in required_roles:
    assert role in roles, f"Missing required role: {role}"
print(f"✓ All required ingredient roles present")

# Test 6: Test role inference
print(f"\nTest 6: Testing ingredient role inference...")
test_ingredients = {
    'chicken': 'protein',
    'garlic': 'aromatic',
    'lemon': 'acid',
    'olive_oil': 'fat',
    'rice': 'starch',
    'carrot': 'vegetable',
    'basil': 'herb',
    'cumin': 'spice'
}
for ing_name, expected_role in test_ingredients.items():
    inferred = infer_ingredient_role(ing_name)
    print(f"  {ing_name} -> {inferred} (expected: {expected_role})")
    assert inferred == expected_role, f"Role mismatch for {ing_name}"
print(f"✓ Role inference working correctly")

# Test 7: Verify cuisine template structure
print(f"\nTest 7: Checking cuisine template structure...")
italian = get_cuisine_template('italian')
assert italian is not None, "Italian template should exist"
assert hasattr(italian, 'base_ingredients'), "Should have base_ingredients"
assert hasattr(italian, 'common_proteins'), "Should have common_proteins"
assert hasattr(italian, 'cooking_methods'), "Should have cooking_methods"
assert hasattr(italian, 'flavor_profile'), "Should have flavor_profile"
print(f"✓ Italian template structure verified")
print(f"  Base ingredients: {italian.base_ingredients}")
print(f"  Cooking methods: {italian.cooking_methods}")

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
