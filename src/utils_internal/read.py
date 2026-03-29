"""
DEPRECATED: This script uses hardcoded paths and old dataset structure.
Use training/build_dataset_final.py instead for current pipeline.
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)

excel_filename = "../../data/raw/TechManualQA_Dataset.csv" 
df = pd.read_csv(excel_filename)
print(f"Loaded {len(df)} Q&A pairs from CSV.")

df_test_batch = df.head(5)
print(f"Running test batch on {len(df_test_batch)} rows...")

results = []

for index, row in tqdm(df_test_batch.iterrows(), total=df_test_batch.shape[0]):
    question = str(row['question_text'])
    answer = str(row['gt_answer_snippet'])
    
    full_text = f"<|user|>\n{question}\n<|assistant|>\n{answer}"
    
    inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    input_ids = inputs.input_ids[0]
    target_index = len(input_ids) - 1
    
    while target_index > 0:
        tok_id = input_ids[target_index]
        tok_str = tokenizer.decode(tok_id).strip()
        
        if any(char.isalnum() for char in tok_str) and '</s>' not in tok_str:
            break
        target_index -= 1

    # 6. Extract Vector at the Target Index
    last_hidden_state = outputs.hidden_states[-1][0, target_index, :]
    
    # Convert to list (2048 floats)
    vector_list = last_hidden_state.tolist()
    
    # 7. Store Data
    results.append({
        "Question": question,
        "Last_Token": tokenizer.decode(input_ids[target_index]),
        "Vector": vector_list  
    })


print("Formatting data for CSV...")
final_df = pd.DataFrame(results)

vector_df = pd.DataFrame(final_df['Vector'].to_list(), columns=[f"v_{i}" for i in range(2048)])

output_df = pd.concat([final_df[['Question', 'Last_Token']], vector_df], axis=1)

output_filename = "../../data/processed/hallucination_training_data.csv"
output_df.to_csv(output_filename, index=False)

print(f"✅ Success! Saved {len(output_df)} rows to {output_filename}")