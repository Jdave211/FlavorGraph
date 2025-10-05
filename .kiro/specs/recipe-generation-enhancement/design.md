# Design Document

## Overview

This design extends the FlavorGraph system to enable AI-powered recipe generation while preserving its existing ingredient pairing and chemical analysis capabilities. The architecture builds upon the existing FlavorGraph2Vec embeddings, compound flavor profiles, and ingredient relationship graph to create a comprehensive recipe generation system.

The design follows a modular approach where new recipe generation components integrate seamlessly with existing data preparation and training pipelines. The system will use the pre-trained FlavorGraph embeddings as a knowledge base to ensure generated recipes are grounded in chemical and culinary science.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FlavorGraph System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  FlavorGraph2Vec │────────▶│  Graph Embeddings│          │
│  │   (Existing)     │         │   (300D vectors) │          │
│  └──────────────────┘         └──────────────────┘          │
│           │                            │                     │
│           │                            │                     │
│           ▼                            ▼                     │
│  ┌─────────────────────────────────────────────┐            │
│  │     Enhanced Training Data Generator        │            │
│  ├─────────────────────────────────────────────┤            │
│  │  • Ingredient Knowledge (existing)          │            │
│  │  • Flavor Analysis (existing)               │            │
│  │  • Substitution Pairs (existing)            │            │
│  │  • Recipe Generation (NEW)                  │            │
│  │  • Recipe Instructions (NEW)                │            │
│  │  • Multi-Cuisine Recipes (NEW)              │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │      Fine-tuned Language Model              │            │
│  │    (LoRA/QLoRA on base LLM)                 │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │      Recipe Generation Engine               │            │
│  ├─────────────────────────────────────────────┤            │
│  │  • Prompt Constructor                       │            │
│  │  • Chemical Validator                       │            │
│  │  • Quality Evaluator                        │            │
│  │  • Substitution Handler                     │            │
│  └─────────────────────────────────────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Request (cuisine, dietary, creativity)
    │
    ▼
Recipe Generation Engine
    │
    ├──▶ Prompt Constructor ──▶ Fine-tuned LLM
    │                                │
    │                                ▼
    │                          Raw Recipe Output
    │                                │
    ├──▶ Chemical Validator ◀────────┘
    │    (checks flavor balance)
    │                │
    │                ▼
    └──▶ Quality Evaluator
         (coherence, novelty, structure)
                    │
                    ▼
              Final Recipe
```

## Components and Interfaces

### 1. Enhanced Training Data Generator

**Purpose:** Extends existing `prepare_training_data.py` to generate recipe-specific training examples.

**New Classes:**

```python
class RecipeDataGenerator:
    """Generates structured recipe training examples"""
    
    def __init__(self, embeddings, nodes_df, edges_df, compound_df):
        self.embeddings = embeddings
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.compound_df = compound_df
        self.cuisine_templates = self._load_cuisine_templates()
        self.cooking_techniques = self._load_cooking_techniques()
    
    def generate_recipe_examples(self, num_recipes: int = 5000) -> List[Dict]:
        """Generate complete recipe training examples"""
        pass
    
    def create_recipe_from_seed_ingredients(self, 
                                           seed_ingredients: List[str],
                                           cuisine: str = None) -> Dict:
        """Create recipe starting from seed ingredients"""
        pass
    
    def generate_recipe_instructions(self, 
                                    ingredients: List[Dict],
                                    cuisine: str,
                                    cooking_method: str) -> List[str]:
        """Generate step-by-step cooking instructions"""
        pass
    
    def balance_flavor_profile(self, ingredients: List[str]) -> Dict[str, float]:
        """Calculate and balance flavor profile for ingredient list"""
        pass
```

**Key Methods:**

- `generate_recipe_examples()`: Creates diverse recipe training data
- `create_recipe_from_seed_ingredients()`: Builds recipes around specific ingredients
- `generate_recipe_instructions()`: Creates realistic cooking steps
- `balance_flavor_profile()`: Ensures chemical/flavor balance

**Integration Point:** Extends `FlavorGraphDataPreparator` class in existing `prepare_training_data.py`

### 2. Recipe Structure Templates

**Purpose:** Define structured formats for different recipe types and cuisines.

**Data Structure:**

```python
@dataclass
class RecipeTemplate:
    cuisine: str
    typical_ingredients: List[str]
    cooking_methods: List[str]  # sauté, roast, simmer, etc.
    flavor_profile: Dict[str, float]  # sweet, salty, umami, etc.
    structure: Dict[str, Any]  # base, protein, aromatics, etc.

@dataclass
class Recipe:
    title: str
    cuisine: str
    servings: int
    prep_time: int  # minutes
    cook_time: int  # minutes
    ingredients: List[Ingredient]
    instructions: List[str]
    flavor_profile: Dict[str, float]
    novelty_score: float
    chemical_coherence: float

@dataclass
class Ingredient:
    name: str
    quantity: float
    unit: str
    node_id: int
    role: str  # 'protein', 'aromatic', 'acid', 'fat', etc.
```

**Cuisine Templates:**

```python
CUISINE_TEMPLATES = {
    'italian': {
        'base_ingredients': ['tomato', 'garlic', 'olive_oil', 'basil'],
        'common_proteins': ['chicken', 'beef', 'seafood'],
        'cooking_methods': ['sauté', 'roast', 'simmer'],
        'flavor_profile': {'umami': 0.6, 'acid': 0.5, 'fat': 0.6, 'aromatic': 0.7}
    },
    'asian': {
        'base_ingredients': ['soy_sauce', 'ginger', 'garlic', 'sesame_oil'],
        'common_proteins': ['chicken', 'pork', 'tofu', 'seafood'],
        'cooking_methods': ['stir_fry', 'steam', 'braise'],
        'flavor_profile': {'umami': 0.8, 'sweet': 0.4, 'aromatic': 0.7, 'heat': 0.5}
    },
    # ... more cuisines
}
```

### 3. Chemical-Aware Recipe Generator

**Purpose:** Use FlavorGraph embeddings to ensure chemical compatibility in generated recipes.

**Class Design:**

```python
class ChemicalAwareRecipeGenerator:
    """Generates recipes with chemical/flavor awareness"""
    
    def __init__(self, embeddings, compound_df, nodes_df):
        self.embeddings = embeddings
        self.compound_df = compound_df
        self.nodes_df = nodes_df
        self.flavor_calculator = FlavorProfileCalculator(compound_df)
    
    def select_complementary_ingredients(self,
                                        seed_ingredients: List[str],
                                        target_flavor: Dict[str, float],
                                        num_ingredients: int = 5) -> List[str]:
        """Select ingredients that complement seeds and match target flavor"""
        pass
    
    def calculate_ingredient_compatibility(self,
                                          ing1: str,
                                          ing2: str) -> float:
        """Calculate compatibility score using embeddings"""
        pass
    
    def balance_recipe_flavors(self,
                              ingredients: List[str],
                              target_profile: Dict[str, float]) -> List[str]:
        """Add/adjust ingredients to achieve target flavor balance"""
        pass
    
    def get_flavor_explanation(self,
                              ingredients: List[str]) -> str:
        """Generate explanation of why ingredients work together"""
        pass
```

**Key Algorithms:**

1. **Ingredient Selection Algorithm:**
   ```
   For each seed ingredient:
       1. Get embedding vector from FlavorGraph2Vec
       2. Find top-k similar ingredients using cosine similarity
       3. Filter by cuisine compatibility
       4. Check flavor profile contribution
       5. Select ingredients that balance target profile
   ```

2. **Flavor Balance Algorithm:**
   ```
   Current_profile = sum(ingredient_flavor_profiles)
   Target_profile = desired_flavor_distribution
   
   For each flavor dimension (sweet, salty, umami, etc.):
       If current < target:
           Find ingredients high in that flavor
           Add to recipe
       If current > target:
           Reduce quantity or find balancing ingredient
   ```

### 4. Recipe Quality Evaluator

**Purpose:** Assess generated recipes for quality, coherence, and feasibility.

**Class Design:**

```python
class RecipeQualityEvaluator:
    """Evaluates quality of generated recipes"""
    
    def __init__(self, embeddings, training_recipes):
        self.embeddings = embeddings
        self.training_recipes = training_recipes
    
    def evaluate_recipe(self, recipe: Recipe) -> Dict[str, float]:
        """Comprehensive recipe evaluation"""
        return {
            'coherence_score': self.calculate_coherence(recipe),
            'novelty_score': self.calculate_novelty(recipe),
            'flavor_balance_score': self.calculate_flavor_balance(recipe),
            'structural_completeness': self.check_structure(recipe),
            'quantity_realism': self.check_quantities(recipe),
            'overall_score': self.calculate_overall_score(recipe)
        }
    
    def calculate_coherence(self, recipe: Recipe) -> float:
        """Check if ingredients work well together using embeddings"""
        pass
    
    def calculate_novelty(self, recipe: Recipe) -> float:
        """Measure how unique the recipe is vs training data"""
        pass
    
    def calculate_flavor_balance(self, recipe: Recipe) -> float:
        """Assess flavor profile distribution"""
        pass
    
    def check_structure(self, recipe: Recipe) -> float:
        """Verify recipe has all required components"""
        pass
    
    def check_quantities(self, recipe: Recipe) -> float:
        """Validate ingredient quantities are realistic"""
        pass
```

**Evaluation Metrics:**

1. **Coherence Score (0-1):**
   - Average pairwise cosine similarity of ingredient embeddings
   - Higher = ingredients work well together

2. **Novelty Score (0-1):**
   - Distance from nearest training recipe in embedding space
   - Higher = more creative/unique

3. **Flavor Balance Score (0-1):**
   - Entropy of flavor profile distribution
   - Optimal balance ≈ 0.7-0.8 (not too flat, not too peaked)

4. **Structural Completeness (0-1):**
   - Binary checks: has title, ingredients, instructions, timing
   - 1.0 = all components present

5. **Quantity Realism (0-1):**
   - Check if quantities fall within reasonable ranges
   - Flag outliers (e.g., 10 cups of salt)

### 5. Recipe Generation API

**Purpose:** Provide clean interface for recipe generation with various parameters.

**API Design:**

```python
class RecipeGenerationAPI:
    """Main API for recipe generation"""
    
    def __init__(self, model_path: str, embeddings_path: str):
        self.model = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer(model_path)
        self.embeddings = self._load_embeddings(embeddings_path)
        self.generator = ChemicalAwareRecipeGenerator(...)
        self.evaluator = RecipeQualityEvaluator(...)
    
    def generate_recipe(self,
                       cuisine: str = None,
                       seed_ingredients: List[str] = None,
                       dietary_restrictions: List[str] = None,
                       servings: int = 4,
                       creativity_level: float = 0.5,
                       target_flavor: Dict[str, float] = None,
                       max_attempts: int = 3) -> Recipe:
        """
        Generate a recipe with specified parameters
        
        Args:
            cuisine: Cuisine type (italian, asian, mexican, etc.)
            seed_ingredients: Starting ingredients to build around
            dietary_restrictions: ['vegetarian', 'gluten_free', etc.]
            servings: Number of servings
            creativity_level: 0.0 (conservative) to 1.0 (experimental)
            target_flavor: Desired flavor profile
            max_attempts: Retry attempts if quality is low
        
        Returns:
            Recipe object with all components
        """
        pass
    
    def substitute_ingredient(self,
                            recipe: Recipe,
                            original_ingredient: str,
                            dietary_restriction: str = None) -> Recipe:
        """Substitute an ingredient while maintaining balance"""
        pass
    
    def explain_recipe(self, recipe: Recipe) -> str:
        """Generate explanation of ingredient choices and chemistry"""
        pass
```

**Request/Response Format:**

```python
# Request
request = {
    "cuisine": "italian",
    "seed_ingredients": ["chicken", "lemon"],
    "dietary_restrictions": [],
    "servings": 4,
    "creativity_level": 0.7,
    "target_flavor": {
        "umami": 0.6,
        "acid": 0.7,
        "aromatic": 0.8
    }
}

# Response
response = {
    "recipe": {
        "title": "Lemon Herb Roasted Chicken with Garlic",
        "cuisine": "italian",
        "servings": 4,
        "prep_time": 15,
        "cook_time": 45,
        "ingredients": [
            {"name": "chicken", "quantity": 1.5, "unit": "lbs", "role": "protein"},
            {"name": "lemon", "quantity": 2, "unit": "whole", "role": "acid"},
            {"name": "garlic", "quantity": 6, "unit": "cloves", "role": "aromatic"},
            {"name": "olive_oil", "quantity": 3, "unit": "tbsp", "role": "fat"},
            {"name": "rosemary", "quantity": 2, "unit": "sprigs", "role": "aromatic"}
        ],
        "instructions": [
            "Preheat oven to 425°F (220°C)",
            "Pat chicken dry and season with salt and pepper",
            "Crush garlic cloves and mix with olive oil, lemon zest, and chopped rosemary",
            "Rub mixture all over chicken, including under the skin",
            "Place lemon halves inside chicken cavity",
            "Roast for 45-50 minutes until internal temperature reaches 165°F",
            "Let rest 10 minutes before carving"
        ],
        "flavor_profile": {
            "umami": 0.55,
            "acid": 0.75,
            "aromatic": 0.85,
            "fat": 0.50,
            "salt": 0.40
        }
    },
    "metadata": {
        "coherence_score": 0.87,
        "novelty_score": 0.45,
        "flavor_balance_score": 0.82,
        "overall_quality": 0.85
    },
    "explanation": "This recipe combines chicken with lemon's citric acid compounds and garlic's sulfur-based aromatics, creating a classic Italian flavor profile. The rosemary adds terpene compounds that complement both the citrus and savory notes."
}
```

## Data Models

### Training Data Format

```json
{
  "instruction": "Generate an Italian recipe using chicken and lemon with a bright, aromatic flavor profile",
  "input": "cuisine: italian, ingredients: [chicken, lemon], flavor: aromatic+acid",
  "output": "**Lemon Herb Roasted Chicken**\n\nServings: 4 | Prep: 15 min | Cook: 45 min\n\n**Ingredients:**\n- 1.5 lbs chicken\n- 2 whole lemons\n- 6 cloves garlic\n- 3 tbsp olive oil\n- 2 sprigs rosemary\n- Salt and pepper to taste\n\n**Instructions:**\n1. Preheat oven to 425°F...\n[full instructions]\n\n**Why this works:** The citric acid in lemon tenderizes the chicken while garlic's allicin compounds create aromatic depth..."
}
```

### Database Schema (for recipe storage)

```python
# If implementing recipe storage/retrieval
class RecipeDatabase:
    recipes: Table[
        id: int,
        title: str,
        cuisine: str,
        ingredients_json: str,  # JSON array
        instructions_json: str,  # JSON array
        flavor_profile_json: str,  # JSON object
        coherence_score: float,
        novelty_score: float,
        created_at: datetime
    ]
    
    ingredient_usage: Table[
        recipe_id: int,
        ingredient_node_id: int,
        quantity: float,
        unit: str,
        role: str
    ]
```

## Error Handling

### Error Types and Handling Strategies

1. **Low Quality Generation:**
   ```python
   if quality_score < threshold:
       # Retry with adjusted parameters
       # Increase temperature for more creativity
       # Or decrease for more conservative output
   ```

2. **Incompatible Ingredients:**
   ```python
   if coherence_score < 0.3:
       # Ingredients don't work together
       # Re-select using stricter similarity thresholds
   ```

3. **Dietary Restriction Violations:**
   ```python
   if any(ing in restricted_ingredients for ing in recipe.ingredients):
       # Filter out restricted ingredients
       # Find substitutes from same category
   ```

4. **Missing Data:**
   ```python
   if ingredient not in embeddings:
       # Fall back to category-based selection
       # Or use string similarity for closest match
   ```

5. **API Timeout:**
   ```python
   @timeout(10)  # 10 second limit
   def generate_recipe(...):
       # If timeout, return cached/template recipe
       # Or return partial result with warning
   ```

## Testing Strategy

### Unit Tests

1. **Training Data Generation:**
   - Test recipe structure validation
   - Test flavor profile calculation
   - Test ingredient compatibility scoring
   - Test cuisine template loading

2. **Recipe Generation:**
   - Test with various cuisine types
   - Test with dietary restrictions
   - Test creativity levels (0.0, 0.5, 1.0)
   - Test seed ingredient handling

3. **Quality Evaluation:**
   - Test coherence calculation
   - Test novelty scoring
   - Test flavor balance assessment
   - Test quantity validation

### Integration Tests

1. **End-to-End Recipe Generation:**
   ```python
   def test_full_recipe_generation():
       api = RecipeGenerationAPI(model_path, embeddings_path)
       recipe = api.generate_recipe(
           cuisine='italian',
           seed_ingredients=['tomato', 'basil'],
           servings=4
       )
       assert recipe.title is not None
       assert len(recipe.ingredients) >= 3
       assert len(recipe.instructions) >= 3
       assert recipe.coherence_score > 0.5
   ```

2. **Substitution Workflow:**
   ```python
   def test_ingredient_substitution():
       recipe = generate_base_recipe()
       modified = api.substitute_ingredient(
           recipe,
           'butter',
           dietary_restriction='vegan'
       )
       assert 'butter' not in [i.name for i in modified.ingredients]
       assert modified.flavor_profile['fat'] > 0.3  # Still has fat
   ```

3. **Multi-Cuisine Generation:**
   ```python
   def test_cuisine_diversity():
       cuisines = ['italian', 'asian', 'mexican', 'indian']
       recipes = [api.generate_recipe(cuisine=c) for c in cuisines]
       # Check that recipes use appropriate ingredients
       assert 'soy_sauce' in recipes[1].ingredients  # Asian
       assert 'cumin' in recipes[3].ingredients  # Indian
   ```

### Evaluation Tests

1. **Human Evaluation Framework:**
   - Generate 50 recipes across cuisines
   - Have human raters score on:
     - Feasibility (can this be cooked?)
     - Flavor logic (do ingredients make sense?)
     - Creativity (is this interesting?)
     - Clarity (are instructions clear?)

2. **Automated Quality Metrics:**
   ```python
   def evaluate_model_quality(num_samples=100):
       recipes = [api.generate_recipe() for _ in range(num_samples)]
       metrics = {
           'avg_coherence': mean([r.coherence_score for r in recipes]),
           'avg_novelty': mean([r.novelty_score for r in recipes]),
           'avg_ingredients': mean([len(r.ingredients) for r in recipes]),
           'structural_completeness': sum([r.is_complete for r in recipes]) / num_samples
       }
       return metrics
   ```

3. **Chemical Validity Tests:**
   ```python
   def test_flavor_balance():
       recipe = api.generate_recipe(target_flavor={'sweet': 0.8})
       actual_sweet = recipe.flavor_profile['sweet']
       assert abs(actual_sweet - 0.8) < 0.2  # Within tolerance
   ```

## Performance Considerations

### Training Performance

- **Data Generation:** ~10-15 minutes for 5,000 recipes (parallelizable)
- **Model Training:** 2-4 hours on GPU for 3 epochs with LoRA
- **Memory:** ~16GB GPU RAM for 7B parameter model with 4-bit quantization

### Inference Performance

- **Recipe Generation:** < 5 seconds per recipe
- **Batch Generation:** ~50 recipes/minute
- **Substitution:** < 2 seconds per substitution

### Optimization Strategies

1. **Caching:**
   - Cache ingredient embeddings in memory
   - Cache common ingredient combinations
   - Cache cuisine templates

2. **Batch Processing:**
   - Generate multiple recipes in parallel
   - Batch embedding lookups

3. **Model Optimization:**
   - Use 4-bit quantization (QLoRA) for inference
   - Optimize prompt length
   - Use KV-cache for faster generation

## Deployment Considerations

### Model Serving

```python
# FastAPI endpoint example
from fastapi import FastAPI
app = FastAPI()

@app.post("/generate_recipe")
async def generate_recipe_endpoint(request: RecipeRequest):
    recipe = api.generate_recipe(
        cuisine=request.cuisine,
        seed_ingredients=request.seed_ingredients,
        dietary_restrictions=request.dietary_restrictions,
        servings=request.servings,
        creativity_level=request.creativity_level
    )
    return recipe
```

### Scaling

- **Horizontal:** Multiple API instances behind load balancer
- **Vertical:** GPU instances for faster inference
- **Caching:** Redis for frequently requested recipes

### Monitoring

- Track generation latency
- Monitor quality scores over time
- Log failed generations for analysis
- Track most requested cuisines/ingredients

## Migration Path

### Phase 1: Data Preparation (Week 1)
- Extend `prepare_training_data.py` with recipe generation
- Create cuisine templates
- Generate 5,000 recipe examples
- Validate data quality

### Phase 2: Model Training (Week 2)
- Combine existing + new training data
- Fine-tune model with LoRA
- Evaluate on held-out test set
- Iterate on data quality if needed

### Phase 3: API Development (Week 3)
- Implement RecipeGenerationAPI
- Add quality evaluation
- Create substitution logic
- Write unit tests

### Phase 4: Integration & Testing (Week 4)
- Integration tests
- Human evaluation
- Performance optimization
- Documentation

This design preserves all existing FlavorGraph functionality while adding powerful recipe generation capabilities grounded in chemical and culinary science.
