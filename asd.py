import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./LeanGPT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Hello.")
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Exit...")
        break
    
    input_text = f"User: {user_input}\nBot:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output_ids = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    bot_response = response.split("Bot:")[-1].strip()
    print("##########################################")
    print(f"Bot: {bot_response}")