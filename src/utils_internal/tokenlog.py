import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, "models", "gemma-2-2b-it")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", dtype=torch.float16
)
model.eval()

# Build a set of every special token ID once at load time.
# This covers <end_of_turn>, <eos>, <pad>, <bos> and any other
# tokens Gemma registers — no hardcoded strings needed.
special_ids = set(tokenizer.all_special_ids)

print("Model loaded. Type 'exit' to quit.\n")


def find_last_meaningful_idx(generated_token_ids: torch.Tensor) -> int:
    """
    Walk backwards through the generated token IDs.
    Return the index of the last token that:
      - is NOT a special token (eos, end_of_turn, pad, bos …)
      - contains at least one alphanumeric character
    Falls back to index 0 if nothing qualifies.
    """
    for i in range(len(generated_token_ids) - 1, -1, -1):
        tok_id  = generated_token_ids[i].item()
        tok_str = tokenizer.decode(tok_id).strip()

        if tok_id in special_ids:          # skip <end_of_turn>, <eos>, etc.
            continue
        if not any(c.isalnum() for c in tok_str):   # skip punctuation / whitespace
            continue

        return i                           # ← first real word token from the end

    return 0                              # fallback


# ── Chat loop ─────────────────────────────────────────────────────────────────
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Exiting.")
        break

    messages = [{"role": "user", "content": user_input}]
    prompt   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            do_sample=False,              # greedy — deterministic for inspection
        )

    input_length   = inputs["input_ids"].shape[1]
    generated_ids  = outputs.sequences[0][input_length:]   # shape: (n_generated,)
    full_response  = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ── Find last meaningful token ────────────────────────────────────────────
    target_index = find_last_meaningful_idx(generated_ids)

    target_token_id  = generated_ids[target_index].item()
    target_token_str = tokenizer.decode(target_token_id)

    # ── Probability of the chosen token at that generation step ───────────────
    # outputs.scores[i] = logits over vocab at generation step i → shape (1, vocab)
    target_prob = F.softmax(outputs.scores[target_index], dim=-1)[0, target_token_id].item()

    # ── Hidden state extraction ───────────────────────────────────────────────
    # outputs.hidden_states is a tuple of length n_generated_tokens.
    # Each element is a tuple of (n_layers + 1) tensors, shape (batch, 1, hidden_dim).
    # [-1] = last layer,  [0] = batch dim,  [-1] = the single position in this step.
    last_hidden = outputs.hidden_states[target_index][-1][0, -1, :]   # shape: (hidden_dim,)
    vector      = last_hidden.float().cpu().tolist()

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n--- Response ---\n{full_response}\n")
    print(f"Target token : '{target_token_str}'  (index {target_index} of {len(generated_ids)} generated)")
    print(f"Probability  : {target_prob:.4f}")
    print(f"Vector dim   : {len(vector)}")
    print(f"Vector : {vector[:]}")
    print("\n" + "=" * 50 + "\n")