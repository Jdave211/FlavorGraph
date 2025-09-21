import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_base_model():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompts = [
        "What is garlic?",
        "Tell me about tomatoes.",
        "How do you cook pasta?",
        "What are the health benefits of spinach?",
        "Describe the taste of chocolate."
    ]
    
    for prompt in prompts:
        print(f"\n🔍 Prompt: {prompt}")
        
        # Try different generation parameters
        inputs = tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
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
    test_base_model()
