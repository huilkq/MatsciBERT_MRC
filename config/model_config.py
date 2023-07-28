import torch

model_type = "bert"
loss_type = "ce"
epochs = 10  # 训练轮次
batch_size = 16  # 每次取的大小
lr = 3e-5  # bert学习率
crf_lr = 1e-4  # 条件随机场学习率
adam_epsilon = 1e-8  # Epsilon for AdamW optimizer
weight_decay = 0.01  # 权重衰减
warm_up_ratio = 0.1  # 预热学习率
max_seq_length = 256  # tokens最大长度


labels_BC5CDR = ["O", "B-Disease", "I-Disease"]
labels_JNLPBA = ["O", "B-protein", "I-protein", "B-DNA", "I-DNA", "B-cell_type", "I-cell_type", "B-cell_line",
                 "I-cell_line", "B-RNA", "I-RNA"]
labels_Metal = ["O", "B-MAT", "I-MAT", "B-SPL", "I-SPL", "B-DSC", "I-DSC", "B-PRO", "I-PRO", "B-APL", "I-APL",
                "B-CMT", "I-CMT", "B-SMT", "I-SMT"]
labels_little = ["O", "B-Metal", "I-Metal", "B-Phenolic", "I-Phenolic", "B-Embellish", "I-Embellish", "B-MOP", "I-MOP",
                 "B-APL", "I-APL",
                 "B-MOC", "I-MOC"]
labels = labels_Metal
labels_to_ids = ""
ids_to_labels = ""

# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT = 'bert-base-cased'
PUBMEBERT = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
BLUEBERT = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
MATSCIBERT = 'rachen/matscibert-squad-accelerate'
SCIBERT = 'D:\pythonProject\BIONER_ER\pre_models\scibert_scivocab_uncased'
BIOBERT = 'D:\pythonProject\BIONER_ER\pre_models\\biobert-base-cased-v1.2'
MODEL = MATSCIBERT
FILE_NAME = 'D:\pythonProject\MatsciBERT_MRC\datasets\Metal'
output_dir = "D:\pythonProject\MatsciBERT_MRC\outputs\cner_output"
