import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# KoBERT 모델/토크나이저 로드 (Hugging Face)
# KoBERT는 'skt/kobert-base-v1'을 사용합니다.
MODEL_NAME = 'skt/kobert-base-v1'

# 설정 상수
MAX_LEN = 128  # 입력 시퀀스 최대 길이
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
NUM_CLASSES = 15  # 클래스 개수: 0부터 14까지 15개
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용할 장치: {DEVICE}")

# --- 1. 데이터셋 준비 (예시 데이터 로드 및 전처리) ---

# 실제 데이터 파일을 로드하는 부분입니다. (예: CSV 파일)
try:
    # 가정: data.csv 파일에 title_clean, comment_clean, label_id 컬럼이 있습니다.
    data = pd.read_csv("your_data.csv") # 실제 파일 경로로 변경하세요.
    
    # 데이터셋 구성 (title_clean, comment_clean 두 개의 텍스트 입력)
    data['text_input'] = data['title_clean'] + " [SEP] " + data['comment_clean'] 
    
    # 학습/검증 데이터 분리
    train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)

except FileNotFoundError:
    print("경고: 'your_data.csv' 파일을 찾을 수 없습니다. 예시 더미 데이터를 사용합니다.")
    # 파일이 없는 경우를 위한 더미 데이터 생성
    train_data = {
        'title_clean': ['안녕하세요', 'KoBERT', '학습을', '시작합니다', '파이썬'],
        'comment_clean': ['분류 모델입니다', 'Fine-tuning 예제', 'CUDA 사용 준비', '다중 클래스', '코드 작성'],
        'label_id': [0, 14, 7, 3, 11]
    }
    train_df = pd.DataFrame(train_data)
    val_df = train_df.copy()
    train_df['text_input'] = train_df['title_clean'] + " [SEP] " + train_df['comment_clean']
    val_df['text_input'] = val_df['title_clean'] + " [SEP] " + val_df['comment_clean']


# 1-1. 커스텀 데이터셋 클래스 정의
class KoBERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.sentences = df['text_input'].tolist()
        self.labels = df['label_id'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 두 개의 입력 텍스트를 '[SEP]' 토큰으로 연결하여 BERT의 입력 형식에 맞춥니다.
        # title_clean과 comment_clean을 이미 " [SEP] "로 연결하여 text_input에 저장했습니다.
        text = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 데이터셋 및 데이터로더 생성
train_dataset = KoBERTDataset(train_df, tokenizer, MAX_LEN)
val_dataset = KoBERTDataset(val_df, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# --- 2. 모델 정의 (분류층 추가) ---
class KoBERTClassifier(torch.nn.Module):
    def __init__(self, bert, num_classes):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        # KoBERT의 hidden_size는 768입니다.
        self.classifier = torch.nn.Linear(768, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # KoBERT의 출력: last_hidden_state, pooler_output
        # pooler_output (BERT의 [CLS] 토큰 출력)을 분류에 사용합니다.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # pooler_output (batch_size, 768)
        pooled_output = outputs.pooler_output 
        
        logits = self.classifier(pooled_output)
        return logits

# 모델 로드 및 장치 이동
bert_model = BertModel.from_pretrained(MODEL_NAME)
model = KoBERTClassifier(bert_model, NUM_CLASSES)
model.to(DEVICE)


# --- 3. 학습 설정 ---

# 옵티마이저 및 스케줄러
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 손실 함수 (다중 분류의 경우 CrossEntropyLoss)
if 'label_id' in train_df.columns and len(train_df) > 0:
    
    # 2. 역빈도에 기반하여 가중치를 계산합니다.
    full_weights = np.zeros(NUM_CLASSES)
    # value_counts()를 사용해 각 클래스 인덱스(label_id)의 개수를 얻습니다.
    for idx, count in train_df['label_id'].value_counts().items():
        # 데이터 수가 0이 아닌 경우에만 가중치를 계산합니다 (1/count).
        if count > 0:
            full_weights[idx] = 1.0 / count
    
    # 3. 가중치 정규화 (선택 사항: 가중치 합이 클래스 개수가 되도록)
    if (full_weights > 0).sum() > 0:
        full_weights = full_weights / full_weights.sum() * (full_weights > 0).sum() 
    
    # 4. 텐서 변환 및 손실 함수에 적용
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
    print("✅ 클래스 불균형 보정 (Class Weighting)이 손실 함수에 적용되었습니다.")
else:
    # 훈련 데이터프레임이 비어있거나 'label_id' 컬럼이 없는 경우
    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
    print("기본 CrossEntropyLoss가 적용되었습니다.")

# [클래스 불균형 보정 로직 삽입 끝]

# --- 4. 학습 함수 정의 ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # 1. Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # 2. Backward
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 3. Optimize & Update
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# --- 5. 평가 함수 정의 ---
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


# --- 6. 학습 루프 실행 ---
print("\n--- 학습 시작 ---")
best_accuracy = 0

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
    print('-' * 10)

    # 학습
    train_acc, train_loss = train_epoch(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        DEVICE,
        scheduler
    )

    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

    # 검증
    val_acc, val_loss = eval_model(
        model,
        val_dataloader,
        loss_fn,
        DEVICE
    )

    print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')

    # 모델 저장 (가장 좋은 성능의 모델)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_kobert_model.bin')
        best_accuracy = val_acc
        print("-> Best model 저장 완료.")

print("\n--- 학습 완료 ---")
