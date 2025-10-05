"""
Recipe data structures and templates for FlavorGraph recipe generation.

This module defines the core data structures for representing recipes, ingredients,
and cuisine-specific templates used in the recipe generation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IngredientRole(Enum):
    """Categories for ingredient roles in recipes"""
    PROTEIN = "protein"
    AROMATIC = "aromatic"
    ACID = "acid"
    FAT = "fat"
    STARCH = "starch"
    VEGETABLE = "vegetable"
    HERB = "herb"
    SPICE = "spice"
    LIQUID = "liquid"
    SWEETENER = "sweetener"
    DAIRY = "dairy"
    OTHER = "other"


class CookingTechnique(Enum):
    """Common cooking techniques"""
    SAUTE = "sauté"
    ROAST = "roast"
    GRILL = "grill"
    STEAM = "steam"
    BRAISE = "braise"
    SIMMER = "simmer"
    BOIL = "boil"
    BAKE = "bake"
    FRY = "fry"
    STIR_FRY = "stir_fry"
    BLANCH = "blanch"
    POACH = "poach"
    BROIL = "broil"
    SMOKE = "smoke"
    MARINATE = "marinate"
    PICKLE = "pickle"
    FERMENT = "ferment"


# List of cooking techniques for easy access
COOKING_TECHNIQUES = [technique.value for technique in CookingTechnique]


@dataclass
class Ingredient:
    """Represents a single ingredient in a recipe"""
    name: str
    quantity: float
    unit: str
    node_id: Optional[int] = None
    role: str = IngredientRole.OTHER.value
    notes: Optional[str] = None
    
    def __str__(self) -> str:
        """Format ingredient for display"""
        return f"{self.quantity} {self.unit} {self.name}"


@dataclass
class Recipe:
    """Represents a complete recipe with all components"""
    title: str
    cuisine: str
    servings: int
    prep_time: int  # minutes
    cook_time: int  # minutes
    ingredients: List[Ingredient]
    instructions: List[str]
    flavor_profile: Dict[str, float] = field(default_factory=dict)
    novelty_score: float = 0.0
    chemical_coherence: float = 0.0
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    @property
    def total_time(self) -> int:
        """Total time in minutes"""
        return self.prep_time + self.cook_time
    
    def __str__(self) -> str:
        """Format recipe for display"""
        output = [
            f"# {self.title}",
            f"\n**Cuisine:** {self.cuisine}",
            f"**Servings:** {self.servings}",
            f"**Prep Time:** {self.prep_time} min | **Cook Time:** {self.cook_time} min",
            f"\n## Ingredients\n"
        ]
        for ing in self.ingredients:
            output.append(f"- {ing}")
        
        output.append("\n## Instructions\n")
        for i, step in enumerate(self.instructions, 1):
            output.append(f"{i}. {step}")
        
        return "\n".join(output)


@dataclass
class RecipeTemplate:
    """Template defining characteristics of a cuisine type"""
    cuisine: str
    base_ingredients: List[str]
    common_proteins: List[str]
    common_vegetables: List[str]
    common_aromatics: List[str]
    cooking_methods: List[str]
    flavor_profile: Dict[str, float]
    typical_herbs_spices: List[str]
    characteristic_techniques: List[str]
    description: str = ""


# Cuisine templates defining characteristics of different culinary traditions
CUISINE_TEMPLATES = {
    'italian': RecipeTemplate(
        cuisine='Italian',
        base_ingredients=['olive_oil', 'garlic', 'tomato', 'basil', 'parmesan'],
        common_proteins=['chicken', 'beef', 'pork', 'seafood', 'veal'],
        common_vegetables=['tomato', 'zucchini', 'eggplant', 'bell_pepper', 'spinach'],
        common_aromatics=['garlic', 'onion', 'shallot'],
        cooking_methods=['sauté', 'roast', 'simmer', 'bake', 'grill'],
        flavor_profile={
            'umami': 0.6,
            'acid': 0.5,
            'fat': 0.6,
            'aromatic': 0.7,
            'sweet': 0.3,
            'salt': 0.5
        },
        typical_herbs_spices=['basil', 'oregano', 'rosemary', 'thyme', 'parsley', 'sage'],
        characteristic_techniques=['sauté', 'simmer', 'roast'],
        description='Mediterranean cuisine emphasizing fresh ingredients, olive oil, and herbs'
    ),
    
    'asian': RecipeTemplate(
        cuisine='Asian',
        base_ingredients=['soy_sauce', 'ginger', 'garlic', 'sesame_oil', 'rice'],
        common_proteins=['chicken', 'pork', 'tofu', 'seafood', 'duck'],
        common_vegetables=['bok_choy', 'mushroom', 'carrot', 'bell_pepper', 'scallion'],
        common_aromatics=['ginger', 'garlic', 'scallion', 'lemongrass'],
        cooking_methods=['stir_fry', 'steam', 'braise', 'boil', 'fry'],
        flavor_profile={
            'umami': 0.8,
            'sweet': 0.4,
            'aromatic': 0.7,
            'heat': 0.5,
            'salt': 0.6,
            'acid': 0.4
        },
        typical_herbs_spices=['ginger', 'star_anise', 'five_spice', 'cilantro', 'sesame'],
        characteristic_techniques=['stir_fry', 'steam', 'braise'],
        description='Diverse Asian cuisines with emphasis on umami, balance, and quick cooking'
    ),
    
    'mexican': RecipeTemplate(
        cuisine='Mexican',
        base_ingredients=['chili_pepper', 'cumin', 'lime', 'cilantro', 'tomato'],
        common_proteins=['chicken', 'beef', 'pork', 'beans', 'seafood'],
        common_vegetables=['tomato', 'onion', 'bell_pepper', 'corn', 'avocado'],
        common_aromatics=['onion', 'garlic', 'chili_pepper'],
        cooking_methods=['grill', 'braise', 'simmer', 'fry', 'roast'],
        flavor_profile={
            'heat': 0.7,
            'acid': 0.6,
            'aromatic': 0.6,
            'umami': 0.5,
            'salt': 0.5,
            'sweet': 0.3
        },
        typical_herbs_spices=['cumin', 'cilantro', 'oregano', 'chili_powder', 'paprika'],
        characteristic_techniques=['grill', 'braise', 'simmer'],
        description='Bold flavors with chili peppers, lime, and complex spice blends'
    ),
    
    'indian': RecipeTemplate(
        cuisine='Indian',
        base_ingredients=['cumin', 'turmeric', 'ginger', 'garlic', 'ghee'],
        common_proteins=['chicken', 'lamb', 'lentils', 'chickpeas', 'paneer'],
        common_vegetables=['tomato', 'onion', 'potato', 'cauliflower', 'spinach'],
        common_aromatics=['ginger', 'garlic', 'onion', 'curry_leaves'],
        cooking_methods=['simmer', 'sauté', 'roast', 'fry', 'steam'],
        flavor_profile={
            'aromatic': 0.9,
            'heat': 0.6,
            'umami': 0.5,
            'fat': 0.6,
            'sweet': 0.4,
            'acid': 0.4
        },
        typical_herbs_spices=['cumin', 'coriander', 'turmeric', 'garam_masala', 'cardamom', 'cinnamon'],
        characteristic_techniques=['simmer', 'sauté', 'roast'],
        description='Complex spice blends with layered aromatics and rich sauces'
    ),
    
    'mediterranean': RecipeTemplate(
        cuisine='Mediterranean',
        base_ingredients=['olive_oil', 'lemon', 'garlic', 'tomato', 'feta'],
        common_proteins=['chicken', 'lamb', 'seafood', 'chickpeas', 'beef'],
        common_vegetables=['tomato', 'cucumber', 'eggplant', 'zucchini', 'bell_pepper'],
        common_aromatics=['garlic', 'onion', 'shallot'],
        cooking_methods=['grill', 'roast', 'sauté', 'bake', 'simmer'],
        flavor_profile={
            'acid': 0.6,
            'aromatic': 0.7,
            'fat': 0.6,
            'umami': 0.5,
            'salt': 0.5,
            'sweet': 0.3
        },
        typical_herbs_spices=['oregano', 'mint', 'dill', 'parsley', 'thyme', 'za_atar'],
        characteristic_techniques=['grill', 'roast', 'sauté'],
        description='Fresh, healthy cuisine with olive oil, citrus, and fresh herbs'
    ),
    
    'french': RecipeTemplate(
        cuisine='French',
        base_ingredients=['butter', 'cream', 'wine', 'shallot', 'thyme'],
        common_proteins=['chicken', 'beef', 'duck', 'seafood', 'pork'],
        common_vegetables=['mushroom', 'leek', 'carrot', 'potato', 'asparagus'],
        common_aromatics=['shallot', 'garlic', 'onion', 'leek'],
        cooking_methods=['sauté', 'braise', 'roast', 'poach', 'bake'],
        flavor_profile={
            'fat': 0.7,
            'umami': 0.6,
            'aromatic': 0.6,
            'acid': 0.4,
            'salt': 0.5,
            'sweet': 0.3
        },
        typical_herbs_spices=['thyme', 'tarragon', 'parsley', 'bay_leaf', 'herbes_de_provence'],
        characteristic_techniques=['sauté', 'braise', 'poach'],
        description='Refined techniques with butter, cream, and wine-based sauces'
    ),
    
    'thai': RecipeTemplate(
        cuisine='Thai',
        base_ingredients=['fish_sauce', 'lime', 'chili', 'coconut_milk', 'lemongrass'],
        common_proteins=['chicken', 'shrimp', 'pork', 'tofu', 'seafood'],
        common_vegetables=['bell_pepper', 'thai_basil', 'bean_sprouts', 'eggplant', 'bamboo_shoots'],
        common_aromatics=['lemongrass', 'galangal', 'garlic', 'shallot', 'kaffir_lime'],
        cooking_methods=['stir_fry', 'simmer', 'grill', 'steam', 'boil'],
        flavor_profile={
            'heat': 0.7,
            'acid': 0.7,
            'sweet': 0.6,
            'aromatic': 0.8,
            'umami': 0.6,
            'salt': 0.5
        },
        typical_herbs_spices=['thai_basil', 'cilantro', 'mint', 'chili', 'galangal'],
        characteristic_techniques=['stir_fry', 'simmer', 'grill'],
        description='Balance of hot, sour, sweet, and salty with aromatic herbs'
    ),
    
    'japanese': RecipeTemplate(
        cuisine='Japanese',
        base_ingredients=['soy_sauce', 'mirin', 'sake', 'dashi', 'rice'],
        common_proteins=['fish', 'tofu', 'chicken', 'pork', 'seafood'],
        common_vegetables=['daikon', 'mushroom', 'seaweed', 'edamame', 'cabbage'],
        common_aromatics=['ginger', 'garlic', 'scallion', 'shiso'],
        cooking_methods=['steam', 'grill', 'simmer', 'fry', 'poach'],
        flavor_profile={
            'umami': 0.9,
            'sweet': 0.4,
            'salt': 0.5,
            'aromatic': 0.5,
            'acid': 0.3,
            'fat': 0.3
        },
        typical_herbs_spices=['ginger', 'wasabi', 'shiso', 'sesame', 'nori'],
        characteristic_techniques=['steam', 'grill', 'simmer'],
        description='Subtle, umami-rich cuisine emphasizing fresh ingredients and presentation'
    ),
    
    'middle_eastern': RecipeTemplate(
        cuisine='Middle Eastern',
        base_ingredients=['olive_oil', 'tahini', 'lemon', 'cumin', 'chickpeas'],
        common_proteins=['lamb', 'chicken', 'beef', 'chickpeas', 'lentils'],
        common_vegetables=['tomato', 'cucumber', 'eggplant', 'onion', 'bell_pepper'],
        common_aromatics=['garlic', 'onion', 'shallot'],
        cooking_methods=['grill', 'roast', 'simmer', 'bake', 'fry'],
        flavor_profile={
            'aromatic': 0.8,
            'acid': 0.5,
            'fat': 0.6,
            'umami': 0.5,
            'sweet': 0.4,
            'salt': 0.5
        },
        typical_herbs_spices=['cumin', 'coriander', 'sumac', 'za_atar', 'mint', 'parsley'],
        characteristic_techniques=['grill', 'roast', 'simmer'],
        description='Aromatic spices, grilled meats, and fresh herbs with tahini and yogurt'
    ),
    
    'american': RecipeTemplate(
        cuisine='American',
        base_ingredients=['butter', 'garlic', 'onion', 'tomato', 'cheese'],
        common_proteins=['beef', 'chicken', 'pork', 'turkey', 'bacon'],
        common_vegetables=['potato', 'corn', 'tomato', 'lettuce', 'onion'],
        common_aromatics=['garlic', 'onion'],
        cooking_methods=['grill', 'roast', 'fry', 'bake', 'smoke'],
        flavor_profile={
            'fat': 0.7,
            'umami': 0.6,
            'salt': 0.6,
            'sweet': 0.5,
            'aromatic': 0.4,
            'acid': 0.3
        },
        typical_herbs_spices=['black_pepper', 'paprika', 'garlic_powder', 'thyme', 'rosemary'],
        characteristic_techniques=['grill', 'roast', 'fry'],
        description='Hearty, comfort-focused cuisine with grilled meats and bold flavors'
    ),
    
    'spanish': RecipeTemplate(
        cuisine='Spanish',
        base_ingredients=['olive_oil', 'garlic', 'tomato', 'paprika', 'saffron'],
        common_proteins=['chicken', 'seafood', 'pork', 'chorizo', 'beef'],
        common_vegetables=['tomato', 'bell_pepper', 'onion', 'potato', 'artichoke'],
        common_aromatics=['garlic', 'onion', 'shallot'],
        cooking_methods=['sauté', 'simmer', 'grill', 'roast', 'fry'],
        flavor_profile={
            'aromatic': 0.7,
            'umami': 0.6,
            'fat': 0.6,
            'acid': 0.4,
            'salt': 0.5,
            'sweet': 0.3
        },
        typical_herbs_spices=['paprika', 'saffron', 'parsley', 'bay_leaf', 'thyme'],
        characteristic_techniques=['sauté', 'simmer', 'grill'],
        description='Regional diversity with emphasis on seafood, olive oil, and paprika'
    ),
    
    'caribbean': RecipeTemplate(
        cuisine='Caribbean',
        base_ingredients=['lime', 'chili', 'coconut', 'allspice', 'thyme'],
        common_proteins=['chicken', 'pork', 'seafood', 'goat', 'beans'],
        common_vegetables=['plantain', 'yam', 'bell_pepper', 'tomato', 'okra'],
        common_aromatics=['garlic', 'onion', 'scallion', 'ginger'],
        cooking_methods=['grill', 'braise', 'fry', 'simmer', 'smoke'],
        flavor_profile={
            'heat': 0.7,
            'aromatic': 0.7,
            'sweet': 0.5,
            'acid': 0.6,
            'umami': 0.4,
            'salt': 0.5
        },
        typical_herbs_spices=['allspice', 'thyme', 'scotch_bonnet', 'cilantro', 'jerk_seasoning'],
        characteristic_techniques=['grill', 'braise', 'fry'],
        description='Tropical flavors with jerk spices, citrus, and coconut influences'
    )
}


# Ingredient role mapping helpers
INGREDIENT_ROLE_KEYWORDS = {
    IngredientRole.PROTEIN: ['chicken', 'beef', 'pork', 'fish', 'tofu', 'lamb', 'turkey', 'shrimp', 'seafood', 'duck', 'veal'],
    IngredientRole.AROMATIC: ['garlic', 'onion', 'ginger', 'shallot', 'leek', 'scallion', 'lemongrass', 'galangal'],
    IngredientRole.ACID: ['lemon', 'lime', 'vinegar', 'tomato', 'wine', 'yogurt'],
    IngredientRole.FAT: ['oil', 'butter', 'ghee', 'cream', 'coconut_milk', 'tahini', 'avocado'],
    IngredientRole.STARCH: ['rice', 'pasta', 'potato', 'bread', 'noodle', 'quinoa', 'couscous'],
    IngredientRole.VEGETABLE: ['carrot', 'broccoli', 'spinach', 'kale', 'zucchini', 'eggplant', 'bell_pepper', 'mushroom'],
    IngredientRole.HERB: ['basil', 'parsley', 'cilantro', 'thyme', 'rosemary', 'oregano', 'mint', 'dill'],
    IngredientRole.SPICE: ['cumin', 'paprika', 'turmeric', 'cinnamon', 'coriander', 'cardamom', 'pepper', 'chili'],
    IngredientRole.LIQUID: ['water', 'stock', 'broth', 'wine', 'sake', 'mirin'],
    IngredientRole.SWEETENER: ['sugar', 'honey', 'maple_syrup', 'agave'],
    IngredientRole.DAIRY: ['cheese', 'milk', 'cream', 'yogurt', 'butter', 'sour_cream']
}


def infer_ingredient_role(ingredient_name: str) -> str:
    """
    Infer the role of an ingredient based on its name.
    
    Args:
        ingredient_name: Name of the ingredient
        
    Returns:
        Role as string (from IngredientRole enum)
    """
    ingredient_lower = ingredient_name.lower().replace(' ', '_')
    
    for role, keywords in INGREDIENT_ROLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in ingredient_lower:
                return role.value
    
    return IngredientRole.OTHER.value


def get_cuisine_template(cuisine: str) -> Optional[RecipeTemplate]:
    """
    Get the template for a specific cuisine type.
    
    Args:
        cuisine: Name of the cuisine (case-insensitive)
        
    Returns:
        RecipeTemplate if found, None otherwise
    """
    cuisine_key = cuisine.lower().replace(' ', '_')
    return CUISINE_TEMPLATES.get(cuisine_key)


def list_available_cuisines() -> List[str]:
    """
    Get list of all available cuisine types.
    
    Returns:
        List of cuisine names
    """
    return [template.cuisine for template in CUISINE_TEMPLATES.values()]
