import json
import torch
from dataclasses import dataclass
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from datasets import Dataset 



@dataclass
class TrainingConfig:
    output_dir: str="./Model"               # Kimenet mappa
    overwrite_output_dir: bool = True       # Felülírás engedélyezése
    num_train_epochs: int = 5               # Epoch-ok száma
    per_device_train_batch_size: int = 2    # Batch méret (GPU függő)
    per_device_eval_batch_size: int = 2     # Eval batch méret
    warmup_steps: int = 500                 # Warmup lépések
    logging_steps: int = 100                # Log gyakoriság
    save_steps: int = 500                   # Mentés gyakorisága
    eval_strategy: str = "steps"            # Kiértékelési stratégia
    eval_steps: int = 500                   # Kiértékelés gyakorisága
    learning_rate: float = 2e-5             # Tanulási ráta
    weight_decay: float = 0.01              # L2 regularizáció
    fp16: bool = True                       # Mixed precision (GPU)
    dataloader_num_workers: int = 2         # Adatbetöltő szálak
    load_best_model_at_end: bool = True


    def to_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir = self.output_dir,
            overwrite_output_dir = self.overwrite_output_dir,
            num_train_epochs = self.num_train_epochs,
            per_device_train_batch_size = self.per_device_train_batch_size,
            per_device_eval_batch_size = self.per_device_eval_batch_size,
            warmup_steps = self.warmup_steps,
            logging_steps = self.logging_steps,
            save_steps = self.save_steps,
            eval_strategy  = self.eval_strategy,
            eval_steps = self.eval_steps,
            learning_rate = self.learning_rate,
            weight_decay = self.weight_decay,
            fp16 = self.fp16,
            dataloader_num_workers = self.dataloader_num_workers,
            load_best_model_at_end = self.load_best_model_at_end
        )


## Train Class ##
class Train():
    def __init__(self, model_name, tokenizer_name, tokenizer, model, data_file):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.model = model
        self.data_file = data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_model()

    def _setup_model(self):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)
            self.model = GPTNeoForCausalLM.from_pretrained(self.model_name)
            device = self.device

            print(f"Model name: {self.model_name}")
            print(f"Tokenizer name: {self.tokenizer_name}")
            print(f"Device: {device}")

            print("\nThe model has loaded successfully.")

        except Exception as e:
            print(e)


    def data_processing(self) -> str:
        corpus = None

        with open(self.data_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            
            corpus_lines = []
            for item in data:
                text = item["text"]
                corpus_lines.append(text)

            corpus = "\n".join(corpus_lines)

        return corpus
    

    def tokenize_corpus(self):
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.resize_token_embeddings(len(self.tokenizer))

        with open(self.data_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    
        texts = [item["text"] for item in data]

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        return tokens



    def prepare_dataset(self):
        tokens = self.tokenize_corpus()
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        dataset = []
        for i in range(input_ids.size(0)):
            dataset.append({
                "input_ids": input_ids[i],
                "attention_mask": attention_mask[i],
                "labels": input_ids[i]
            })

        hf_dataset = Dataset.from_list(dataset)
        split = hf_dataset.train_test_split(test_size=0.1)
        return split["train"], split["test"]


    def train_model(self):
        train_dataset, eval_dataset = self.prepare_dataset()
        training_config = TrainingConfig()
        training_args = training_config.to_training_args()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )

        torch.cuda.empty_cache()
        trainer.train()
        self.save_model()



    def save_model(self):
        self.model.save_pretrained("./Model")
        self.tokenizer.save_pretrained("./Model")
        print("Model was saved. (./Model)")


if __name__ == "__main__":
    ## Model settings ##
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer_name = "EleutherAI/gpt-neo-125M"
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    data_file = "data.json"


    train = Train(model_name, tokenizer_name, tokenizer, model, data_file)
    train.train_model()
        
