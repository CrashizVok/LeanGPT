from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./Model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

while True:
    prompt = input("\n>>>>> ")
    if prompt.lower() == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{completion}")
