import torch
from transformers import GPTNeoForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path="./Model/Version 1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def chat(tokenizer, model, device, prompt, max_length=150, temperature=0.95, top_p=0.9):
    style_prompt = (
        "Speaketh in the style of William Shakespeare:\n"
        f"User: {prompt}\n"
        "AI:"
    )

    tokens = tokenizer(style_prompt, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # <-- ez a kulcs
            max_length=input_ids.shape[1] + max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text[len(style_prompt):].strip()


if __name__ == "__main__":
    tokenizer, model, device = load_model_and_tokenizer()

    print("🎭 Hail, noble friend! Speak thy mind (type 'exit' to depart).\n")

    while True:
        prompt = input("Thou: ")
        if prompt.strip().lower() in ["exit", "quit", "farewell"]:
            print("AI: Fare thee well, sweet interlocutor.")
            break

        reply = chat(tokenizer, model, device, prompt)
        print(f"AI: {reply}\n")
