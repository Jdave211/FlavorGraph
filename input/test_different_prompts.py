import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def test_different_prompts():
    model_path = "../output/efficient_eval_flavorgraph/"
    base_model = "microsoft/DialoGPT-medium"
    
    print("🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, model_path)
    
    print("✅ Model loaded successfully!")
    
    # Test with different types of prompts
    test_prompts = [
        "What is garlic?",
        "Tell me about tomatoes.",
        "How do you cook pasta?",
        "What are the health benefits of spinach?",
        "Describe the taste of chocolate.",
        "What ingredients go well with chicken?",
        "Explain the difference between herbs and spices.",
        "What is the nutritional value of broccoli?",
        "How do you make a salad?",
        "What are the benefits of eating fish?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔍 Test {i}: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
            
            # Count exclamation marks
            exclamation_count = response.count('!')
            print(f"Exclamation marks: {exclamation_count}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_different_prompts()
