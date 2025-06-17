import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def generate_next_word(model, tokenizer, prompt, max_length=50):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).unsqueeze(0)

    generated_ids = torch.cat([input_ids[0], next_token_id])
    generated_text = tokenizer.decode(generated_ids)

    # Külön a következő szó, a prompt után
    next_word = tokenizer.decode(next_token_id)

    return generated_text, next_word


if __name__ == "__main__":
    model_path = "./Model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path)

    prompt = input("Adj meg egy kezdő mondatot: ")
    full_text, next_word = generate_next_word(model, tokenizer, prompt)

    print(f"Teljes szöveg: {full_text}")
    print(f"Következő szó: {next_word}")
