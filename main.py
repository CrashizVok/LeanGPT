import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_path = "./Model/checkpoint-10500"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    while True:
        prompt = input(">>>> ")
        output = generate_text(model, tokenizer, prompt)
        print(f"SecretGPT: {output}")
