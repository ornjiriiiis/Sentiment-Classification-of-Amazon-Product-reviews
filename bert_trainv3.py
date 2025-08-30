import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
)
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from torch.nn import CrossEntropyLoss

# ========================================
# STEP 1: LOAD AND CLEAN DATA 
# ========================================
data_dir = "/home/ornjira/internship/amazon_dataset/"

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def map_sentiment(star):
    if star in [1, 2]:
        return 0  # bad
    elif star == 3:
        return 1  # normal
    else:
        return 2  # good

def load_single_tsv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip',
                         engine='python', encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_balanced_samples(directory, samples_per_class=5000):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            path = os.path.join(directory, filename)
            df = load_single_tsv(path)
            if df is None or 'star_rating' not in df or 'review_body' not in df:
                continue
            df = df.dropna(subset=['star_rating', 'review_body'])
            df['review_headline'] = df['review_headline'].fillna('').apply(clean_text)
            df['review_body'] = df['review_body'].fillna('').apply(clean_text)
            df = df[df['review_body'].str.len() > 10]
            df['sentiment'] = df['star_rating'].astype(int).apply(map_sentiment)
            balanced_df = (
                df.groupby('sentiment', group_keys=False)
                .apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42))
            )
            dataframes.append(balanced_df)
    return pd.concat(dataframes, ignore_index=True)

df_sample = load_balanced_samples(data_dir, samples_per_class=5000)
headlines = df_sample['review_headline'].tolist()
bodies = df_sample['review_body'].tolist()
labels = df_sample['sentiment'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
token_lens = [len(tokenizer.encode(h + ' ' + b)) for h, b in zip(headlines, bodies)]
print(f"Avg token length: {np.mean(token_lens):.2f}, Max token length: {np.max(token_lens)}")

# ========================================
# STEP 2: TOKENIZATION & DATASET
# ========================================
class AmazonDataset(Dataset):
    def __init__(self, headlines, bodies, labels, tokenizer, max_len=512):
        self.texts = [h + ' [SEP] ' + b for h, b in zip(headlines, bodies)]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

head_train, head_temp, body_train, body_temp, y_train, y_temp = train_test_split(
    headlines, bodies, labels, test_size=0.3, stratify=labels, random_state=42
)
head_val, head_test, body_val, body_test, y_val, y_test = train_test_split(
    head_temp, body_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_dataset = AmazonDataset(head_train, body_train, y_train, tokenizer)
val_dataset = AmazonDataset(head_val, body_val, y_val, tokenizer)
test_dataset = AmazonDataset(head_test, body_test, y_test, tokenizer)

# ========================================
# STEP 3: MODEL & TRAINING
# ========================================
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
)
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from torch.nn import CrossEntropyLoss

# ========================================
# STEP 1: LOAD AND CLEAN DATA 
# ========================================
data_dir = "/home/ornjira/internship/amazon_dataset/"

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def map_sentiment(star):
    if star in [1, 2]:
        return 0  # bad
    elif star == 3:
        return 1  # normal
    else:
        return 2  # good

def load_single_tsv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip',
                         engine='python', encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_balanced_samples(directory, samples_per_class=5000):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            path = os.path.join(directory, filename)
            df = load_single_tsv(path)
            if df is None or 'star_rating' not in df or 'review_body' not in df:
                continue
            df = df.dropna(subset=['star_rating', 'review_body'])
            df['review_headline'] = df['review_headline'].fillna('').apply(clean_text)
            df['review_body'] = df['review_body'].fillna('').apply(clean_text)
            df = df[df['review_body'].str.len() > 10]
            df['sentiment'] = df['star_rating'].astype(int).apply(map_sentiment)
            balanced_df = (
                df.groupby('sentiment', group_keys=False)
                .apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42))
            )
            dataframes.append(balanced_df)
    return pd.concat(dataframes, ignore_index=True)

df_sample = load_balanced_samples(data_dir, samples_per_class=5000)
headlines = df_sample['review_headline'].tolist()
bodies = df_sample['review_body'].tolist()
labels = df_sample['sentiment'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
token_lens = [len(tokenizer.encode(h + ' ' + b)) for h, b in zip(headlines, bodies)]
print(f"Avg token length: {np.mean(token_lens):.2f}, Max token length: {np.max(token_lens)}")

# ========================================
# STEP 2: TOKENIZATION & DATASET
# ========================================
class AmazonDataset(Dataset):
    def __init__(self, headlines, bodies, labels, tokenizer, max_len=512):
        self.texts = [h + ' [SEP] ' + b for h, b in zip(headlines, bodies)]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

head_train, head_temp, body_train, body_temp, y_train, y_temp = train_test_split(
    headlines, bodies, labels, test_size=0.3, stratify=labels, random_state=42
)
head_val, head_test, body_val, body_test, y_val, y_test = train_test_split(
    head_temp, body_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_dataset = AmazonDataset(head_train, body_train, y_train, tokenizer)
val_dataset = AmazonDataset(head_val, body_val, y_val, tokenizer)
test_dataset = AmazonDataset(head_test, body_test, y_test, tokenizer)

# ========================================
# STEP 3: CUSTOM TRAINER + METRICS
# ========================================
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device)
        loss_fct = CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, digits=4, output_dict=True)
    return {"f1": report["weighted avg"]["f1-score"]}

# ========================================
# STEP 4: MULTI-RUN EXPERIMENT LOOP
# ========================================
weight_sets = [
    [1.0, 1.0, 1.0],
    [1.0, 1.5, 1.0],
    [1.0, 2.0, 1.0],
    [0.8, 2.5, 0.8]
]

for i, weights in enumerate(weight_sets):
    run_id = f"run_weights_{i+1}_w{weights[0]}_{weights[1]}_{weights[2]}"
    output_dir = f"./results_{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Training with weights: {weights} ===")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)

    trainer = CustomTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_first_step=True
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights_tensor
    )
    print(model)
    trainer.train()

    # === Evaluation ===
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    true_labels = [example['labels'].item() for example in test_dataset]

    # Save plots + metrics
    def save_eval_results():
        print(f"Saving evaluation for {run_id}")

        # Misclassified samples
        texts_test = [h + ' [SEP] ' + b for h, b in zip(head_test, body_test)]
        confidences = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        max_conf = np.max(confidences, axis=1)

        results_df = pd.DataFrame({
            'text': texts_test,
            'true_label': true_labels,
            'predicted_label': preds,
            'confidence': max_conf
        })

        label_map = {0: 'bad', 1: 'normal', 2: 'good'}
        results_df['true_label_name'] = results_df['true_label'].map(label_map)
        results_df['predicted_label_name'] = results_df['predicted_label'].map(label_map)

        misclassified = results_df[results_df['true_label'] != results_df['predicted_label']]
        misclassified.to_csv(f"{output_dir}/misclassified_all.csv", index=False)
        misclassified[misclassified['true_label'] == 1].to_csv(f"{output_dir}/misclassified_normal.csv", index=False)

        # Confusion Matrix
        cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bad", "normal", "good"])
        disp.plot(cmap="Blues", values_format='d')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()

        # Classification Report
        report_dict = classification_report(true_labels, preds, digits=4, labels=[0, 1, 2],
                                            target_names=["bad", "normal", "good"], output_dict=True)
        pd.DataFrame(report_dict).transpose().to_csv(f"{output_dir}/classification_report.csv")

        # Loss & F1 Curve
        logs = trainer.state.log_history
        train_loss = [x['loss'] for x in logs if 'loss' in x and 'epoch' in x]
        eval_loss = [x['eval_loss'] for x in logs if 'eval_loss' in x and 'epoch' in x]
        eval_f1 = [x['eval_f1'] for x in logs if 'eval_f1' in x and 'epoch' in x]
        epochs = [x['epoch'] for x in logs if 'eval_loss' in x]

        min_len = min(len(train_loss), len(epochs))
        plt.figure()
        plt.plot(epochs[:min_len], train_loss[:min_len], label='Train Loss')
        plt.plot(epochs, eval_loss, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train vs Validation Loss")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_curve.png")
        plt.close()

        plt.figure()
        plt.plot(epochs, eval_f1, label='Validation F1 Score', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Validation F1 Score Over Epochs")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/f1_score_curve.png")
        plt.close()

    save_eval_results()

    # Save model and tokenizer
    model.save_pretrained(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/model")
