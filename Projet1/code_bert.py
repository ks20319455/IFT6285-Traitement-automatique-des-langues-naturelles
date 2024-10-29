import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Charger le tokenizer et le modèle BERT préentraîné
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9)  # 9 classes

# Charger les jeux de données depuis les fichiers CSV
dataset = load_dataset('csv', data_files={'train': 'train-devinesqui.csv', 'test': 'test-devinesqui.csv'}, delimiter=';')

# Fonction de tokenisation, incluant les titres, abstracts, et labels
def tokenize_function(examples):
    texts = [title + " " + abstract for title, abstract in zip(examples['title'], examples['abstract'])]
    # Retourner les textes tokenisés et inclure les labels
    return tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt')

# Appliquer la tokenisation au jeu de données
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Renommer la colonne 'classe' en 'labels' pour que Trainer reconnaisse les labels
tokenized_datasets = tokenized_datasets.rename_column("classe", "labels")

# S'assurer que les labels sont bien au format entier
def convert_labels(examples):
    label_map = {label: idx for idx, label in enumerate(sorted(set(examples['labels'])))}
    examples['labels'] = [label_map[label] for label in examples['labels']]
    return examples

# Appliquer la conversion des labels sur les datasets train et test
tokenized_datasets = tokenized_datasets.map(convert_labels, batched=True)

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Évalue après chaque époque
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=30,  # Sauvegarde après chaque 500 pas (ajustez selon votre besoin)
    load_best_model_at_end=True
)

# Fonction facultative pour calculer des métriques personnalisées
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Initialiser Trainer pour gérer l'entraînement et l'évaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,  # Passer le tokenizer
    compute_metrics=compute_metrics  # Facultatif : ajouter des métriques personnalisées
)

checkpoint = None
if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir):
    checkpoint = training_args.output_dir

# Lancer l'entraînement
trainer.train(resume_from_checkpoint=checkpoint)

# Évaluer le modèle sur le test set
trainer.evaluate()