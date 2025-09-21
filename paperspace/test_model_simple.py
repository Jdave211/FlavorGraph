import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def test_model():
    model_path = "../output/efficient_eval_flavorgraph/"
    base_model = "microsoft/DialoGPT-medium"
    
    print("🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    
    print("✅ Model loaded successfully!")
    
    # Test with simple generation
    test_prompts = [
        "What is garlic?",
        "Tell me about tomatoes.",
        "How do you cook pasta?"
    ]
    
    for prompt in test_prompts:
        print(f"\n🔍 Testing: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

if __name__ == "__main__":
    test_model()
