import torch
from load_model import loadmodel 

tokenizer, model = loadmodel()
messages = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break
        
    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print(f"\nModel: {response}\n")
    
    messages.append({"role": "assistant", "content": response})

    token_ids = tokenizer.encode(response, add_special_tokens=False)
    print(f"Exact token count: {len(token_ids)}")