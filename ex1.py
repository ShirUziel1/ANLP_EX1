import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score
from pathlib import Path

base_dir = Path(__file__).resolve().parent
import wandb

# os.environ["WANDB_DISABLED"] = "true"
class BertMRPCTrainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(42)
        np.random.seed(42)

        self.dataset = load_dataset("glue", "mrpc")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenized_dataset = self._tokenize_dataset()
        self.model = None

    def _tokenize_dataset(self):
        dataset = self.dataset
        tokenizer = self.tokenizer

        tokenized = dataset.map(lambda x: tokenizer(
            x["sentence1"],
            x["sentence2"],
            truncation=True,
            padding=False
        ), batched=True)

        if self.args.max_train_samples != -1:
            tokenized["train"] = tokenized["train"].select(range(self.args.max_train_samples))
        if self.args.max_eval_samples != -1:
            tokenized["validation"] = tokenized["validation"].select(range(self.args.max_eval_samples))
        if self.args.max_predict_samples != -1:
            tokenized["test"] = tokenized["test"].select(range(self.args.max_predict_samples))

        return tokenized

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    def train(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        training_args = TrainingArguments(
            output_dir="output",
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            learning_rate=self.args.lr,
            num_train_epochs=self.args.num_train_epochs,
            logging_steps=1,
            run_name=f"ep{self.args.num_train_epochs}_lr{self.args.lr}_bs{self.args.batch_size}",
            save_strategy="no",
            report_to="wandb",
            logging_dir=str(base_dir / "logs")
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        eval_acc = eval_results["eval_accuracy"]
        model_save_path = base_dir / f"saved_models/ep{self.args.num_train_epochs}_lr{self.args.lr}_bs{self.args.batch_size}"
        model_save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

        # Log results
        line = (
            f"epoch_num: {self.args.num_train_epochs}, lr: {self.args.lr}, "
            f"batch_size: {self.args.batch_size}, eval_acc: {eval_acc:.4f}"
        )
        with open(base_dir / "res.txt", "a") as f:
            f.write(line + "\n")
        print("Appended validation result to res.txt")


    def predict(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        self.model.eval()

        pred_args = TrainingArguments(
            output_dir="output_predict",
            per_device_eval_batch_size=self.args.batch_size,
            do_train=False,
            report_to=None
        )
        trainer = Trainer(
            model=self.model,
            args=pred_args,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )

        test_tok = self.tokenized_dataset["test"]
        raw_test = self.dataset["test"]
        preds_output = trainer.predict(test_tok)
        preds = np.argmax(preds_output.predictions, axis=1)

        # Write out predictions
        pred_filename = "predictions.txt"
        with open(pred_filename, "w", encoding="utf-8") as f:
            for ex, p in zip(raw_test, preds):
                f.write(f"{ex['sentence1']}###{ex['sentence2']}###{p}\n")

        print(f"Saved predictions to: {pred_filename}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(
        project="ANLP-ex1",
        name=f"epoch_num_{int(args.num_train_epochs)}_lr_{args.lr}_batch_size_{args.batch_size}",
        config={
            "epochs": args.num_train_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }
    )
    trainer = BertMRPCTrainer(args)

    if args.do_train:
        trainer.train()

    if args.do_predict:
        if args.model_path is None:
            print("No model_path provided, selecting best model from res.txt...")
            args.model_path = trainer.get_best_model_path()
            if args.model_path is None:
                raise ValueError("No valid model_path found in res.txt.")
            print(f"Using best model: {args.model_path}")
        trainer.predict()


if __name__ == "__main__":
    main()
