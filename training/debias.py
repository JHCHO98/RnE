import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def csv_to_list_of_dicts(file_path):
    try:
        df = pd.read_csv(file_path)
        try:
            df['label'] = df['label'].astype(int)
        except ValueError:
            print("경고: 'label' 열에 정수로 변환할 수 없는 값이 포함되어 있습니다.")
        result_list = df.to_dict('records')
        return result_list
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"오류: 파일이 비어 있습니다: {file_path}")
        return None
    except Exception as e:
        print(f"처리 중 예상치 못한 오류 발생: {e}")
        return None

data_list = csv_to_list_of_dicts("silver_data_fixed.csv")

# ==========================================
# 1. 설정 및 하이퍼파라미터 (개선)
# ==========================================
CONFIG = {
    'model_name': "jhgan/ko-sbert-nli",
    'num_classes': 3,
    'hidden_dim': 768,
    'nhead': 8,
    'num_layers': 2,
    'batch_size': 16,
    'epochs': 50,                 # 30 → 50 (충분한 학습 시간)
    'learning_rate': 1e-5,        # 5e-6 → 1e-5 (2배 증가)
    'weight_decay': 0.03,         # 0.05 → 0.03 (약간 완화)
    'dropout': 0.5,               # 0.6 → 0.5 (약간 완화)
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Current Device: {CONFIG['device']}")

# ==========================================
# 2. 데이터셋 클래스
# ==========================================
class YouTubeBiasDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        texts = [item['title'], item['comment']]
        label = item['label']
        return texts, label

def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return texts, labels

# ==========================================
# 3. 개선된 모델 아키텍처
# ==========================================
class BiasAnalyzer(nn.Module):
    def __init__(self, config):
        super(BiasAnalyzer, self).__init__()
        self.device = config['device']
        
        # A. Backbone: KR-SBERT
        print(f"Loading SBERT model: {config['model_name']}...")
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # *** BERT 파라미터 상위 4개 레이어 학습 (표현력 향상) ***
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  # 기본적으로 모두 Freeze
            # 상위 4개 레이어 + Pooler 학습
            if 'pooler' in name or any(f'encoder.layer.{i}' in name for i in [8, 9, 10, 11]):
                param.requires_grad = True
        
        print(f"✓ BERT 학습 가능 파라미터 수: {sum(p.numel() for p in self.bert.parameters() if p.requires_grad):,}")
        
        # B. Interaction: Transformer Encoder with Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'], 
            nhead=config['nhead'], 
            dropout=config['dropout'],  # Dropout 추가
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_layers']
        )
        
        # C. 개선된 Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.LayerNorm(config['hidden_dim'] // 2),  # Batch Norm 대신 Layer Norm
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, config['num_classes'])
        )

        self.to(self.device)

    def get_sbert_embeddings(self, flat_texts):
        inputs = self.tokenizer(
            flat_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.bert(**inputs)
        
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def forward(self, batch_texts):
        batch_size = len(batch_texts)
        seq_len = 2
        
        flat_texts = [t for video in batch_texts for t in video]
        flat_embeddings = self.get_sbert_embeddings(flat_texts)
        
        context_input = flat_embeddings.view(batch_size, seq_len, -1)
        context_output = self.transformer(context_input)
        video_vector = torch.mean(context_output, dim=1)
        
        logits = self.classifier(video_vector)
        
        return logits, video_vector

# ==========================================
# 4. Early Stopping 클래스
# ==========================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):  # min_delta를 0으로
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
            print(f"  [Early Stopping Counter: {self.counter}/{self.patience}]")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==========================================
# 5. 실행 및 테스트
# ==========================================
if __name__ == "__main__":
    
    if data_list is None or len(data_list) < 5:
        print("오류: CSV 파일 로드에 실패했거나 데이터가 너무 적습니다.")
        exit()

    # --- A. 데이터 분할 ---
    print(f"\n[데이터 분할 시작] 전체 샘플 수: {len(data_list)}개")
    
    train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    if len(train_data) == 0:
        print("\n!!! 치명적 오류: 학습 데이터셋이 0개입니다. !!!")
        exit()

    print(f"  - 학습(Train) 데이터: {len(train_data)}개")
    print(f"  - 검증(Validation) 데이터: {len(val_data)}개")
    print(f"  - 테스트(Test) 데이터: {len(test_data)}개")
    
    # --- 클래스 가중치 계산 (불균형 대응) ---
    print("\n[클래스 분포 분석 및 가중치 계산]")
    train_labels = [item['label'] for item in train_data]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    
    # Inverse Frequency Weighting: weight = total / (n_classes * count)
    class_weights = total_samples / (CONFIG['num_classes'] * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(CONFIG['device'])
    
    print(f"  클래스별 샘플 수: {class_counts}")
    print(f"  클래스별 가중치: {class_weights.cpu().numpy()}")
    print(f"  → 소수 클래스(2번)에 {class_weights[2]:.2f}배 페널티 부여")
    
    # --- B. 데이터 로더 ---
    train_dataset = YouTubeBiasDataset(train_data)
    val_dataset = YouTubeBiasDataset(val_data)
    test_dataset = YouTubeBiasDataset(test_data)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # --- C. 모델 초기화 ---
    model = BiasAnalyzer(CONFIG)
    
    # *** AdamW with Weight Decay (L2 정규화) ***
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # *** Learning Rate Scheduler - Warmup 추가 ***
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(epoch):
        # Warmup: 처음 3 epoch은 학습률을 점진적으로 증가
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # *** Weighted CrossEntropyLoss (클래스 불균형 대응) ***
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    best_val_accuracy = 0.0

    # --- D. 학습 및 검증 ---
    print("\n[Start Training & Validation]")
    
    for epoch in range(CONFIG['epochs']):
        # 1. TRAIN PHASE
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for texts, labels in train_dataloader: 
            labels = labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits, _ = model(texts)
            loss = criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1) 
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0) 
            
            loss.backward()
            
            # *** Gradient Clipping (그래디언트 폭발 방지) ***
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(train_dataloader)
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # 2. VALIDATION PHASE
        model.eval() 
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad(): 
            for texts, labels in val_dataloader:
                labels = labels.to(CONFIG['device'])
                logits, _ = model(texts)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        val_accuracy = val_correct / val_samples if val_samples > 0 else 0.0
        val_loss_avg = val_loss / len(val_dataloader)

        # 현재 학습률 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.2e}")

        # 3. Learning Rate Scheduling
        if epoch < 3:
            warmup_scheduler.step()  # Warmup 단계
        else:
            plateau_scheduler.step(val_loss_avg)  # Plateau 단계
        
        # 4. Best Model 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            MODEL_SAVE_PATH = 'best_debias.bin'
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> [모델 저장] 검증 정확도 {best_val_accuracy:.4f} 달성, {MODEL_SAVE_PATH}에 저장.")
        
        # 5. Early Stopping 체크
        early_stopping(val_loss_avg)
        if early_stopping.early_stop:
            print(f"\n*** Early Stopping at Epoch {epoch+1} ***")
            break
            
    # --- E. 최종 테스트 ---
    print("\n[Start Final Test on Test Set]")
    
    # 최고 성능 모델 로드
    model.load_state_dict(torch.load('best_debias.bin'))
    model.eval()
    
    test_correct = 0
    test_samples = 0
    all_vectors_list = []
    all_titles_list = []
    
    with torch.no_grad():
        test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
        for texts, labels in test_dataloader:
            labels = labels.to(CONFIG['device'])
            logits, vectors = model(texts)
            
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_samples += labels.size(0)
            
            all_vectors_list.append(vectors.cpu().numpy())
            all_titles_list.extend([t[0] for t in texts])

    if len(all_vectors_list) > 0:
        all_vectors = np.concatenate(all_vectors_list, axis=0)
    else:
        all_vectors = np.array([])
    
    test_accuracy = test_correct / test_samples if test_samples > 0 else 0.0
    print(f"\n=== 최종 테스트 결과 ===")
    print(f"Test Accuracy: {test_accuracy:.4f} (Total {test_samples} Samples)")
    
    # --- 클래스별 정확도 분석 (불균형 확인) ---
    print("\n[클래스별 성능 분석]")
    test_labels_list = []
    test_preds_list = []
    
    with torch.no_grad():
        test_dataloader_analysis = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
        for texts, labels in test_dataloader_analysis:
            labels = labels.to(CONFIG['device'])
            logits, _ = model(texts)
            preds = torch.argmax(logits, dim=1)
            
            test_labels_list.extend(labels.cpu().numpy())
            test_preds_list.extend(preds.cpu().numpy())
    
    # 클래스별 정확도 계산
    for class_id in range(CONFIG['num_classes']):
        mask = np.array(test_labels_list) == class_id
        if mask.sum() > 0:
            class_acc = (np.array(test_preds_list)[mask] == class_id).mean()
            print(f"  클래스 {class_id} 정확도: {class_acc:.4f} (샘플 수: {mask.sum()}개)")
        else:
            print(f"  클래스 {class_id}: 테스트 데이터 없음")

    # --- F. 군집화 ---
    if all_vectors.shape[0] >= CONFIG['num_classes']:
        print("\n[Start Clustering on Test Data Embeddings]")
        kmeans = KMeans(n_clusters=CONFIG['num_classes'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_vectors)

        print(f"\n=== 군집화 결과 (총 {all_vectors.shape[0]} 샘플 기준) ===")
        for i, title in enumerate(all_titles_list): 
            print(f"영상 제목: '{title}' -> 군집 ID: {labels[i]}")
    else:
        print(f"\n[군집화 건너뜀] 샘플 수가 군집 수보다 적습니다.")

    print("\n=== 작업 완료 ===")
