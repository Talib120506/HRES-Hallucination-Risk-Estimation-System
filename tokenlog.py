# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)

# prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nwhy did napolean use iphone 17 to win the french revolution\n<|assistant|>\n"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         return_dict_in_generate=True, 
#         output_scores=True  
#     )

# input_length = inputs["input_ids"].shape[1] 
# generated_tokens = outputs.sequences[0][input_length:]

# print(f"Generated text: {tokenizer.decode(generated_tokens, skip_special_tokens=True)}\n")
# print("--- Token Probabilities ---")

# for step, score in enumerate(outputs.scores):
#     probs = F.softmax(score, dim=-1)
#     chosen_token_id = generated_tokens[step]
#     chosen_prob = probs[0, chosen_token_id].item()
#     token_str = tokenizer.decode(chosen_token_id)
#     print(f"Token: '{token_str:<10}' | Probability: {chosen_prob:.4f}")

# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # 1. Load Model
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)

# # 2. Prepare Input
# prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is a CPU?\n<|assistant|>\n"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# # 3. Generate 
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs, 
#         max_new_tokens=256, 
#         return_dict_in_generate=True, 
#         output_scores=True, 
#         output_hidden_states=True
#     )

# input_length = inputs.input_ids.shape[1]
# generated_tokens = outputs.sequences[0][input_length:]
# full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

# # 4. Backtrack to find the last ACTUAL word (Skip EOS and punctuation)
# ignore_list = ['.', ',', '!', '?', '\n', '</s>', '', ' ']
# target_index = len(generated_tokens) - 1

# while target_index > 0:
#     tok_id = generated_tokens[target_index]
#     # .strip() removes leading/trailing spaces from the token string
#     tok_str = tokenizer.decode(tok_id).strip()
    
#     # If the token is not in our ignore list, we found our target word!
#     if tok_str not in ignore_list:
#         break
        
#     target_index -= 1 # Step back one token

# # 5. Extract target data
# target_token_id = generated_tokens[target_index]
# target_token_str = tokenizer.decode(target_token_id)
# target_prob = F.softmax(outputs.scores[target_index], dim=-1)[0, target_token_id].item()

# target_hidden_state = outputs.hidden_states[target_index][-1][0, -1, :]
# vector_list = target_hidden_state.tolist()

# # --- FINAL OUTPUT ---
# print(f"--- Complete Generated Response ---\n{full_response}\n")
# print(f"Target Token Extracted: '{target_token_str}'")
# print(f"Target Token Probability: {target_prob:.4f}\n")
# print("--- Entire Vector (2048 numbers) ---")
# print(vector_list)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load Model (This happens only once outside the loop)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
print("Model loaded successfully! Type 'exit' to quit.\n")

# 2. Start the Continuous Chat Loop
while True:
    # Get user input
    user_input = input("You: ")
    
    # Check for exit command
    if user_input.lower() == 'exit':
        print("Exiting chat...")
        break
        
    # Format the prompt dynamically with the new user input
    prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 3. Generate 
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            return_dict_in_generate=True, 
            output_scores=True, 
            output_hidden_states=True
        )

    # 4. Process Tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[0][input_length:]
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

# 5. Backtrack to find the last ACTUAL word 
    target_index = len(generated_tokens) - 1

    while target_index > 0:
        tok_id = generated_tokens[target_index]
        tok_str = tokenizer.decode(tok_id).strip()
        
        if any(char.isalnum() for char in tok_str) and '</s>' not in tok_str:
            break
            
        target_index -= 1

    # 6. Extract target data
    target_token_id = generated_tokens[target_index]
    target_token_str = tokenizer.decode(target_token_id)
    target_prob = F.softmax(outputs.scores[target_index], dim=-1)[0, target_token_id].item()

    target_hidden_state = outputs.hidden_states[target_index][-1][0, -1, :]
    vector_list = target_hidden_state.tolist()

    # --- FINAL OUTPUT ---
    print(f"\n--- Complete Generated Response ---\n{full_response}\n")
    print(f"Target Token Extracted: '{target_token_str}'")
    print(f"Target Token Probability: {target_prob:.4f}\n")
    print(f"--- Entire Vector ({len(vector_list)} numbers) ---")
    print(vector_list)
    print("\n" + "="*50 + "\n") 