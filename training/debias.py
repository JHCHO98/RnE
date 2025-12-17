import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt # 시각화를 위한 라이브러리 추가

def csv_to_list_of_dicts(file_path):
    try:
        df = pd.read_csv(file_path)
        try:
            # 'label' 열이 없으면 경고 후 더미 데이터 반환
            if 'label' not in df.columns:
                 print(f"경고: 파일에 'label' 열이 없습니다. 더미 데이터를 사용하여 계속합니다.")
                 return None # 더미 데이터 로직을 트리거하기 위해 None 반환
            
            df['label'] = df['label'].astype(int)
        except ValueError:
            print("'label' 열에 정수로 변환할 수 없는 값이 포함되어 있습니다. 문자열로 유지합니다.")
        result_list = df.to_dict('records')
        return result_list
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}. 더미 데이터를 사용하여 계속합니다.")
        return None # 더미 데이터 로직을 트리거하기 위해 None 반환
    except pd.errors.EmptyDataError:
        print(f"오류: 파일이 비어 있습니다: {file_path}. 더미 데이터를 사용하여 계속합니다.")
        return None
    except Exception as e:
        print(f"처리 중 예상치 못한 오류 발생: {e}")
        return None

# 실제 데이터 로드 시도
data_list = csv_to_list_of_dicts("silver_data_fixed.csv")

# 파일 로드 실패 또는 데이터 부족 시 더미 데이터 생성
if data_list is None or len(data_list) < 5:
    print("\n경고: 실제 데이터 로드 실패 또는 데이터 부족. 더미 데이터를 생성합니다.")
    dummy_data = {
        'title': ['영상 제목 A', '제목 B가 더 좋음', '논란의 영상 C', '평화로운 D', '정치 주제 E'] * 20,
        'comment': ['내용이 편향적이다', '이 댓글은 중립이다', '댓글이 공격적이다', '너무 유익하다', '양쪽 의견이 다 있다'] * 20,
        'label': [0, 1, 2, 1, 0] * 20 # 3개 클래스 (0, 1, 2)
    }
    data_list = pd.DataFrame(dummy_data).to_dict('records')
    print(f"[더미 데이터 생성 완료] 총 {len(data_list)}개")


CONFIG = {
    'model_name': "monologg/koelectra-base-v3",
    'num_classes': 3,
    'hidden_dim': 768,
    'nhead': 6,
    'num_layers': 2,
    'batch_size': 4,
    'epochs': 100, # 학습 속도가 느려질 수 있으므로, 실제 학습 시에는 에포크 수를 적절히 조정하세요.
    'learning_rate': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Current Device: {CONFIG['device']}")

class YouTubeBiasDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 'title'과 'comment' 키가 없는 경우에 대한 예외 처리 (더미 데이터가 아닌 경우를 대비)
        texts = [item.get('title', ''), item.get('comment', '')]
        label = item.get('label', 0)
        return texts, label

def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long) # 라벨을 long 타입으로 지정
    return texts, labels

class BiasAnalyzer(nn.Module):
    def __init__(self, config):
        super(BiasAnalyzer, self).__init__()
        self.device = config['device']
        print(f"Loading BERT model: {config['model_name']}...")
        # KoELECTRA 모델 로드
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # Transformer Encoder Layer 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['nhead'],
            batch_first=True,
            # 주의: KoELECTRA는 BERT 계열이므로 TransformerEncoderLayer의 dim_feedforward, dropout 등
            # 매개변수 설정을 기본값(혹은 적절한 값)으로 유지하는 것이 좋습니다.
            dim_feedforward=config['hidden_dim'] * 4, # 일반적인 FFN 크기
            dropout=0.1
        )
        # Transformer Encoder 정의 (문맥 통합 모듈)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # 분류기 (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['hidden_dim'] // 2, config['num_classes'])
        )
        self.to(self.device)

    # 평균 풀링 (Mean Pooling)을 사용하여 텍스트 임베딩을 얻는 함수
    def get_bert_embeddings(self, flat_texts):
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
        
        # 평균 풀링 구현
        # 마스크를 확장하여 패딩 토큰을 제외하고 평균 계산
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, batch_texts):
        # batch_texts: [['title1', 'comment1'], ['title2', 'comment2'], ...]
        batch_size = len(batch_texts)
        seq_len = 2 # title, comment 두 개
        
        # 1. 모든 텍스트를 평탄화하여 KoELECTRA에 입력
        flat_texts = [t for video in batch_texts for t in video]
        flat_embeddings = self.get_bert_embeddings(flat_texts) # (batch_size * 2, hidden_dim)

        # 2. 임베딩을 (batch_size, 2, hidden_dim) 형태로 재구성
        context_input = flat_embeddings.view(batch_size, seq_len, -1)

        # 3. Transformer Encoder를 통해 title과 comment 사이의 문맥 관계를 학습
        context_output = self.transformer(context_input) # (batch_size, 2, hidden_dim)

        # 4. 출력 시퀀스의 평균을 취해 최종 비디오 벡터 생성
        video_vector = torch.mean(context_output, dim=1) # (batch_size, hidden_dim)

        # 5. 최종 벡터를 분류기에 통과시켜 로짓 얻기
        logits = self.classifier(video_vector) # (batch_size, num_classes)
        
        # video_vector는 군집화를 위해 필요하므로 함께 반환
        return logits, video_vector

if __name__ == "__main__":
    if data_list is None or len(data_list) < 5:
        print("오류: 데이터 로드 실패 또는 데이터 부족. 학습을 시작할 수 없습니다.")
        exit()

    print(f"\n[데이터 분할 시작] 전체 샘플 수: {len(data_list)}개")
    # 레이블을 기준으로 계층적 샘플링을 시도합니다. (레이블이 정수형이라고 가정)
    try:
        labels_for_split = [item['label'] for item in data_list]
        train_data, temp_data = train_test_split(
            data_list,
            test_size=0.2,
            random_state=42,
            stratify=labels_for_split # 계층적 샘플링 적용
        )
        temp_labels = [item['label'] for item in temp_data]
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels
        )
    except Exception as e:
        print(f"경고: 계층적 샘플링 실패 ({e}). 일반 샘플링으로 대체합니다.")
        train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


    if len(train_data) == 0:
        print("\n!!! 치명적 오류: 학습 데이터셋(Train Data)이 0개입니다. !!!")
        exit()

    print(f"  - 학습(Train) 데이터: {len(train_data)}개")
    print(f"  - 검증(Validation) 데이터: {len(val_data)}개")
    print(f"  - 테스트(Test) 데이터: {len(test_data)}개")

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

    model = BiasAnalyzer(CONFIG)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0

    # 시각화를 위한 결과 기록 리스트 초기화
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    print("\n[Start Training & Validation]")

    for epoch in range(CONFIG['epochs']):
        # --- Training ---
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
            optimizer.step()
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(train_dataloader)
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # --- Validation ---
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

        # 결과 기록 (시각화를 위해 추가)
        history['train_acc'].append(train_accuracy)
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_accuracy)
        history['val_loss'].append(val_loss_avg)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            MODEL_SAVE_PATH = 'best_debias_roberta1.bin'
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> [모델 저장] 검증 정확도 {best_val_accuracy:.4f} 달성, {MODEL_SAVE_PATH}에 저장.")
            
    # --- Final Test ---
    print("\n[Start Final Test on Test Set]")
    # 학습된 최적 모델 가중치 로드 (선택적)
    try:
        model.load_state_dict(torch.load('best_debias_roberta1.bin'))
        print("-> 최적 모델 가중치 로드 완료.")
    except:
        print("-> 최적 모델 가중치 로드 실패. 마지막 에포크 모델 사용.")
        
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
    print(f"\n=== 최종 테스트 결과 ===\nTest Accuracy: {test_accuracy:.4f} (Total {test_samples} Samples)")

    # --- Clustering ---
    if all_vectors.shape[0] >= CONFIG['num_classes']:
        print("\n[Start Clustering on Test Data Embeddings]")
        kmeans = KMeans(n_clusters=CONFIG['num_classes'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_vectors)

        print(f"\n=== 군집화 결과 (총 {all_vectors.shape[0]} 샘플 기준) ===")
        for i, title in enumerate(all_titles_list):
            print(f"영상 제목: '{title}' -> 군집 ID: {labels[i]}")

    else:
        print(f"\n[군집화 건너뜜] 샘플 수({all_vectors.shape[0]}개)가 군집 수({CONFIG['num_classes']}개)보다 적습니다.")
    
    # --- 시각화 및 PNG 파일 저장 (추가된 부분) ---
    print("\n--- 학습 결과 시각화 및 PNG 저장 ---")

    epochs_range = range(1, len(history['train_loss']) + 1)

    # 그림 (Figure) 객체 생성
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 손실(Loss) 그래프
    ax[0].plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax[0].plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
    ax[0].set_title('Training and Validation Loss ')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    # 2. 정확도(Accuracy) 그래프
    ax[1].plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o')
    ax[1].plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()

    # 그래프를 'debias_training_results.png' 파일로 저장
    try:
        plt.savefig('debias_training_results.png', dpi=300)
        print("✅ 그래프가 'debias_training_results.png' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ 그래프 저장 중 오류 발생: {e}")

    # 화면에 그래프 표시 (Jupyter/Colab 환경에서만 유효)
    # plt.show() # 스크립트 실행 환경에서는 plt.show()를 주석 처리하거나 제거하는 것이 좋습니다.

    print("\n=== 작업 완료 ===")
