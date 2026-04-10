"""
Answer Generator Service
Uses Gemma to generate answers based on PDF context.
"""
import gc
import numpy as np
import torch
from .model_loader import get_llama, get_embedder
from .detection import get_pdf_index

MAX_CTX_CHARS = 1600


def generate_answer(pdf_path, question):
    """
    Generate an answer using Gemma based on retrieved PDF context.
    
    Returns: (answer_string, error_string)
    """
    # Get PDF index and chunks
    cache_data, err = get_pdf_index(pdf_path)
    if err:
        return None, err
    
    index, all_chunks = cache_data["index"], cache_data["chunks"]
    
    # Retrieve top chunks using the question as query
    embedder = get_embedder()
    q_vec = embedder.encode(
        [question], normalize_embeddings=True, device="cuda"
    ).astype(np.float32)
    
    # Get top 5 relevant chunks (increased from 3 for better context coverage)
    distances, indices = index.search(q_vec, min(5, len(all_chunks)))
    
    # Combine top chunks as context
    context_parts = []
    for i in indices[0]:
        if i >= 0:
            context_parts.append(all_chunks[i])
    
    # Use more context (increased from 1600 to 2400 characters)
    context = " ".join(context_parts)[:2400]
    
    # Debug: Print retrieved context
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print(f"RETRIEVED {len(context_parts)} CHUNKS")
    print(f"CONTEXT ({len(context)} chars):")
    print(context[:500] + "..." if len(context) > 500 else context)
    print("="*80 + "\n")
    
    if not context.strip():
        return None, "No relevant context found in the document"
    
    # Generate answer using Gemma
    tokenizer, model = get_llama()
    
    # Create prompt for answer generation
    messages = [
        {
            "role": "user", 
            "content": f"""Based on the following context from a document, please answer the question concisely and accurately.

Context: {context}

Question: {question}

Please provide a direct, factual answer based only on the information in the context above. If the answer cannot be found in the context, say "The information is not available in the provided context."""
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated answer (with special tokens to see the structure)
    generated_text_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debug: Print the full generated text
    print("\n" + "="*80)
    print("GENERATED TEXT (WITH TOKENS):")
    print(repr(generated_text_full[:500]))
    print("\nGENERATED TEXT (CLEAN):")
    print(repr(generated_text[:500]))
    print("="*80 + "\n")
    
    # Extract just the assistant's response
    # Gemma format typically: <start_of_turn>user\nPROMPT<end_of_turn>\n<start_of_turn>model\nRESPONSE<end_of_turn>
    answer = generated_text
    
    # Method 1: Look for model turn in version with special tokens
    if "<start_of_turn>model" in generated_text_full:
        # Extract content between <start_of_turn>model and <end_of_turn>
        model_start = generated_text_full.find("<start_of_turn>model")
        if model_start != -1:
            after_model = generated_text_full[model_start + len("<start_of_turn>model"):]
            end_turn = after_model.find("<end_of_turn>")
            if end_turn != -1:
                answer = after_model[:end_turn].strip()
            else:
                answer = after_model.strip()
    
    # Method 2: If still has prompt text, try splitting by the instruction end
    elif 'say "The information is not available in the provided context."' in generated_text:
        parts = generated_text.split('say "The information is not available in the provided context."')
        if len(parts) > 1:
            answer = parts[-1].strip()
    
    # Method 3: If still contaminated, split by the question itself
    elif question in generated_text and len(answer) > 300:
        # The answer probably starts after the question
        parts = generated_text.split(question)
        if len(parts) > 1:
            # Take everything after the last occurrence of the question
            after_question = parts[-1].strip()
            # Remove common suffixes that might be in the prompt
            for suffix in ["Please provide a direct, factual answer", "based only on the information", "\n\n"]:
                if suffix in after_question:
                    after_question = after_question.split(suffix)[-1].strip()
            if len(after_question) < len(answer) and len(after_question) > 3:
                answer = after_question
    
    # Clean up any remaining artifacts
    answer = answer.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()
    if answer.lower().startswith("model"):
        answer = answer[5:].strip()
    
    # Remove newlines at the start
    answer = answer.lstrip("\n").strip()
    
    print(f"EXTRACTED ANSWER: {repr(answer)}\n")
    
    # Clean up
    del outputs, inputs
    torch.cuda.empty_cache()
    gc.collect()
    
    if not answer or len(answer.strip()) < 5:
        return None, "Failed to generate a meaningful answer"
    
    return answer.strip(), None
