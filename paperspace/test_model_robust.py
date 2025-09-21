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
    
    # Test with very conservative generation parameters
    test_prompts = [
        "What is garlic?",
        "Tell me about tomatoes."
    ]
    
    for prompt in test_prompts:
        print(f"\n🔍 Testing: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        try:
            with torch.no_grad():
                # Use very conservative parameters
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,  # Very short responses
                    temperature=0.1,    # Very low temperature
                    top_p=0.5,         # Very low top_p
                    do_sample=False,   # Use greedy decoding instead
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,  # No repetition penalty
                    no_repeat_ngram_size=0   # No n-gram repetition
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Trying with even more conservative settings...")
            
            # Try with even more conservative settings
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Response (conservative): {response}")
                
            except Exception as e2:
                print(f"Still failing: {e2}")

if __name__ == "__main__":
    test_model()
