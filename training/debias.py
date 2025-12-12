import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np

# ... (csv_to_list_of_dicts, YouTubeBiasDataset, collate_fn 등 데이터 로딩 부분은 기존과 동일) ...
# ... (csv_to_list_of_dicts 함수와 data_list 로딩 코드는 그대로 두세요) ...

# ==========================================
# 1. 설정 및 하이퍼파라미터 (과적합 방지용 수정)
# ==========================================
CONFIG = {
    'model_name': "jhgan/ko-sbert-nli",
    'num_classes': 3,
    'hidden_dim': 768,
    'batch_size': 32,             # 16 -> 32 (배치를 키워 일반화 성능 향상)
    'epochs': 30,                 # 50 -> 30 (Early Stopping 믿고 줄임)
    'lr_backbone': 2e-6,          # BERT는 천천히 학습 (사전지식 보존)
    'lr_head': 5e-4,              # 분류기는 빠르게 학습
    'weight_decay': 0.05,         # L2 정규화 강화
    'dropout': 0.5,
    'patience': 5,                # 15 -> 5 (과적합 징후 보이면 빨리 멈춤)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'label_smoothing': 0.1        # 과신 방지
}

# ==========================================
# 2. 개선된 모델: Attention Pooling 사용 (경량화)
# ==========================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        # weights: (batch_size, seq_len, 1)
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        
        # Weighted Sum -> (batch_size, hidden_dim)
        # 제목과 댓글 중 더 중요한 정보에 가중치를 두어 합침
        output = torch.sum(x * weights, dim=1)
        return output, weights

class BiasAnalyzer(nn.Module):
    def __init__(self, config):
        super(BiasAnalyzer, self).__init__()
        self.device = config['device']
        
        print(f"Loading SBERT model: {config['model_name']}...")
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # [핵심] 과적합 방지를 위해 BERT 파라미터 대부분 동결 (Last Layer만 학습)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            # 마지막 인코더 레이어(layer.11)와 Pooler만 풂
            if 'encoder.layer.11' in name or 'pooler' in name:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"✓ BERT 학습 가능 파라미터 수: {trainable_params:,} (Top Layer Only)")
        
        # [변경] TransformerEncoder 제거 -> Attention Pooling 도입
        self.pooling = AttentionPooling(config['hidden_dim'])
        
        # Classification Head (구조 단순화)
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.GELU(), # ReLU보다 부드러운 활성화 함수
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, config['num_classes'])
        )

        self.to(self.device)

    def get_sbert_embeddings(self, flat_texts):
        # 토크나이징
        inputs = self.tokenizer(
            flat_texts, return_tensors='pt', padding=True, truncation=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        
        # Mean Pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, batch_texts):
        batch_size = len(batch_texts)
        # Flatten: [[제목, 댓글], ...] -> [제목, 댓글, 제목, 댓글 ...]
        flat_texts = [t for video in batch_texts for t in video]
        
        # Embedding
        flat_embeddings = self.get_sbert_embeddings(flat_texts)
        
        # Reshape: (Batch, 2, Hidden)
        seq_embeddings = flat_embeddings.view(batch_size, 2, -1)
        
        # Attention Pooling: 제목과 댓글의 중요도를 계산해 합침
        video_vector, attn_weights = self.pooling(seq_embeddings)
        
        # Classification
        logits = self.classifier(video_vector)
        
        return logits, video_vector

# ==========================================
# 3. Early Stopping (동일)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  [Early Stopping Count: {self.counter}/{self.patience}]")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==========================================
# 4. 실행 및 학습 루프 (차등 학습률 적용)
# ==========================================
if __name__ == "__main__":
    
    # ... (데이터 로드 및 분할 코드는 기존과 동일하게 유지) ...
    # 편의상 여기서는 데이터 분할 코드가 실행되었다고 가정하고 이어갑니다.
    # 위에서 data_list가 로드되어 있어야 합니다.
    
    if data_list is None or len(data_list) < 5:
        print("오류: 데이터 로드 실패")
        exit()

    # 데이터 분할 (기존 코드 재사용)
    train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42, stratify=[d['label'] for d in data_list])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=[d['label'] for d in temp_data])
    
    # 클래스 가중치 계산 (기존 코드 유지)
    train_labels = [item['label'] for item in train_data]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (CONFIG['num_classes'] * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(CONFIG['device'])
    
    # DataLoader 설정
    train_dataset = YouTubeBiasDataset(train_data)
    val_dataset = YouTubeBiasDataset(val_data)
    test_dataset = YouTubeBiasDataset(test_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # 모델 초기화
    model = BiasAnalyzer(CONFIG)
    
    # [핵심] 차등 학습률(Differential Learning Rate) 적용
    # BERT는 천천히, 분류기는 빠르게 학습
    optimizer_grouped_parameters = [
        {'params': model.bert.parameters(), 'lr': CONFIG['lr_backbone']},  # 2e-6
        {'params': model.pooling.parameters(), 'lr': CONFIG['lr_head']},   # 5e-4
        {'params': model.classifier.parameters(), 'lr': CONFIG['lr_head']} # 5e-4
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=CONFIG['weight_decay'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # [핵심] Label Smoothing 적용 (label_smoothing=0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CONFIG['label_smoothing'])
