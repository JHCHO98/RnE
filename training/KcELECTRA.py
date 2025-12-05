import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# KcELECTRA 모델/토크나이저 로드
MODEL_NAME = 'monologg/kcelectra-base-v2022' 

# 설정 상수
MAX_LEN = 128    # 입력 시퀀스 최대 길이
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
NUM_CLASSES = 15 # 클래스 개수: 0부터 14까지 15개
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용할 장치: {DEVICE}")

# --- 1. 데이터셋 준비 (예시 데이터 로드 및 전처리) ---

# 실제 데이터 파일을 로드하는 부분입니다. (예: CSV 파일)
try:
    # 가정: data.csv 파일에 title_clean, comment_clean, label_id 컬럼이 있습니다.
    data = pd.read_csv("your_data.csv") # 실제 파일 경로로 변경하세요.
    
    # --- 핵심 수정 부분 ---
    # 데이터셋 구성: 제목과 하나의 'comment_clean' 컬럼을 [SEP] 토큰으로 연결합니다.
    data['text_input'] = data['title_clean'] + " [SEP] " + data['comment_clean'].fillna('')
    # ------------------
    
    # 학습/검증 데이터 분리
    train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)

except FileNotFoundError:
    print("경고: 'your_data.csv' 파일을 찾을 수 없습니다. 예시 더미 데이터를 사용합니다.")
    # 파일이 없는 경우를 위한 더미 데이터 생성 (단일 comment_clean 컬럼 사용)
    train_data = {
        'title_clean': ['안녕하세요', 'KcELECTRA', '댓글이 중요합니다', '다중 클래스', '코드 작성'],
        'comment_clean': ['영상 리뷰 좋아요', '구어체와 신조어가 많음', '이 모델로 분류합니다', '클래스 15개 충분', '최종적으로 완성'],
        'label_id': [0, 14, 7, 3, 11]
    }
    train_df = pd.DataFrame(train_data)
    val_df = train_df.copy()
    
    # --- 핵심 수정 부분 (더미 데이터) ---
    train_df['text_input'] = train_df['title_clean'] + " [SEP] " + train_df['comment_clean']
    val_df['text_input'] = val_df['title_clean'] + " [SEP] " + val_df['comment_clean']
    # ------------------


# 1-1. 커스텀 데이터셋 클래스 정의 (이전 코드와 동일)
class KCELectraDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.sentences = df['text_input'].tolist()
        self.labels = df['label_id'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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

# KcELECTRA 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 데이터셋 및 데이터로더 생성
train_dataset = KCELectraDataset(train_df, tokenizer, MAX_LEN)
val_dataset = KCELectraDataset(val_df, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# --- 2. 모델 정의 (분류층 추가) (이전 코드와 동일) ---
class KCELectraClassifier(torch.nn.Module):
    def __init__(self, electra, num_classes):
        super(KCELectraClassifier, self).__init__()
        self.electra = electra
        self.classifier = torch.nn.Linear(768, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # last_hidden_state의 첫 번째 토큰([CLS] 토큰)을 사용
        cls_output = outputs[0][:, 0, :] 
        
        logits = self.classifier(cls_output)
        return logits

# 모델 로드 및 장치 이동
electra_model = AutoModel.from_pretrained(MODEL_NAME)
model = KCELectraClassifier(electra_model, NUM_CLASSES)
model.to(DEVICE)

