import json
import torch
from dataclasses import dataclass
from transformers import GPTNeoForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
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
    learning_rate: float = 5e-5             # Tanulási ráta
    weight_decay: float = 0.01              # L2 regularizáció
    fp16: bool = True                       # Mixed precision (GPU)
    dataloader_num_workers: int = 2         # Adatbetöltő szálak
    load_best_model_at_end: bool = True
    gradient_accumulation_steps: int = 4
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

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
            load_best_model_at_end = self.load_best_model_at_end,
            gradient_accumulation_steps = self.gradient_accumulation_steps,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better
        )

## Train Class ##
class Train():
    def __init__(self, model_name, tokenizer_name, data_file):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_file = data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_config = TrainingConfig()
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        
        self._setup_tokenizer()
        self._setup_model()

    def _setup_tokenizer(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        special_tokens = {
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
            "unk_token": "<unk>",
            "sep_token": "<sep>" 
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        print(f"Tokenizer vocabulary size: {len(self.tokenizer)}")

    def _setup_model(self):
        try:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)
            
            print(f"Model name: {self.model_name}")
            print(f"Tokenizer name: {self.tokenizer_name}")
            print(f"Device: {self.device}")
            print("The model has loaded successfully.")

        except Exception as e:
            print(f"Error: {e}")

    def data_processing(self) -> str:
        with open(self.data_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            
            corpus_lines = []
            for item in data:
                text = item["text"]
                corpus_lines.append(text)

            corpus = "\n".join(corpus_lines)

        return corpus
    
    def tokenize_corpus(self):
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
        training_args = self.training_config.to_training_args()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False, 
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        torch.cuda.empty_cache()
        print("Training started...")
        trainer.train()
        self.save_model()

    def save_model(self):
        self.model.save_pretrained("./Model")
        self.tokenizer.save_pretrained("./Model")
        with open("./Model/training_config.json", "w") as f:
            json.dump(self.training_config.__dict__, f, indent=4)
            
        print("Model and training_config saved to ./Model")

if __name__ == "__main__":
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer_name = "EleutherAI/gpt-neo-125M"
    data_file = "data.json"

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train = Train(model_name, tokenizer_name, data_file)
    train.train_model()
    