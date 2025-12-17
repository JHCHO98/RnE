// Dashboard.js
// React 및 useState 훅을 가져옵니다.
import React, { useState } from 'react';
// 스타일시트를 가져옵니다.
import './Dashboard.css';

// 차트 라이브러리 (예: Chart.js 또는 Recharts)를 가져왔다고 가정합니다.
// import { Scatter } from 'react-chartjs-2'; 

// 1. 가상 데이터 정의
const userData = {
    username: 'User_Alpha',
    categories: [
        { name: '정치 및 시사', percent: 45 },
        { name: 'IT / 테크', percent: 20 },
        { name: '경제 / 금융', percent: 15 },
        { name: '게임', percent: 10 },
        { name: '기타', percent: 10 },
    ],
    biasScore: 0.35, // +0.35 (약간 보수적)
    historyBias: 0.35,
    recoBias: 0.65,
    biasDrift: 0.30,
    filterBubble: {
        level: 4, // 5단계 중 4단계
        criteria: [
            { name: '동일 성향 추천 비율', value: '75%', met: true },
            { name: '반대 성향 노출 비율', value: '8%', met: true },
            { name: '편향 이동도', value: '+0.30', met: true },
        ]
    }
};

// 메인 대시보드 컴포넌트
function Dashboard() {
    // 7번 '연구 보고서' 섹션의 열림/닫힘 상태를 관리
    const [isReportOpen, setIsReportOpen] = useState(false);

    // 6번 '시뮬레이션' 버튼 클릭 이벤트 핸들러
    const handleSimulateClick = () => {
        alert('시뮬레이션 결과: 48시간 내 추천 편향이 +0.65에서 +0.15 (중립)로 이동할 것으로 예측됩니다.');
    };

    // 4번 필터 버블 레벨을 시각적으로 표현
    const renderBubbleMeter = (level) => {
        let bubbles = '';
        for (let i = 0; i < 5; i++) {
            bubbles += (i < level) ? '●' : '○';
        }
        return bubbles;
    };

    // 편향 점수(-1 ~ +1)를 퍼센티지(0% ~ 100%)로 변환 (CSS left 속성용)
    const getBiasPosition = (score) => ((score + 1) / 2) * 100 + '%';

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <h1>유튜브 알고리즘 확증 편향 분석 대시보드</h1>
                <p>경기북과학고 21기 R&E 1학년 팀 | 분석 대상: {userData.username}</p>
            </header>

            <main className="dashboard-grid">
                
                {/* 1. 사용자 프로필 요약 */}
                <section className="card profile-card">
                    <h2>1️⃣ 사용자 프로필 요약</h2>
                    <strong>주요 관심 카테고리 (TOP 5)</strong>
                    <ul className="category-list">
                        {userData.categories.map(cat => (
                            <li key={cat.name}>
                                <span>{cat.name}</span>
                                <span className="percent">{cat.percent}%</span>
                            </li>
                        ))}
                    </ul>
                    <strong>시청 기록 기반 정치 성향: +{userData.biasScore}</strong>
                    <div className="bias-scale-container">
                        <div className="bias-scale-bar">
                            <div 
                                className="bias-marker" 
                                style={{ left: getBiasPosition(userData.biasScore) }}
                                title={`편향 점수: +${userData.biasScore}`}
                            ></div>
                        </div>
                        <div className="bias-labels">
                            <span>진보 (-1.0)</span>
                            <span>중립 (0.0)</span>
                            <span>보수 (+1.0)</span>
                        </div>
                    </div>
                </section>

                {/* 2. 편향 탐지 결과 */}
                <section className="card bias-result-card">
                    <h2>2️⃣ 알고리즘 편향 탐지 결과</h2>
                    <table className="bias-table">
                        <tbody>
                            <tr>
                                <td>🔵 시청 기록 (History)</td>
                                <td className="score-history">+{userData.historyBias}</td>
                            </tr>
                            <tr>
                                <td>🔴 추천 영상 (Reco)</td>
                                <td className="score-reco">+{userData.recoBias}</td>
                            </tr>
                            <tr>
                                <td>🔥 편향 이동도 (Drift)</td>
                                <td className="score-drift">+{userData.biasDrift}</td>
                            </tr>
                        </tbody>
                    </table>
                    <div className="conclusion-box danger">
                        <strong>결론: 알고리즘이 편향을 '강화'하고 있습니다.</strong>
                        <p>알고리즘이 사용자의 기존 성향보다 더 강한 편향의 콘텐츠를 추천하고 있습니다.</p>
                    </div>
                </section>

                {/* 3. 핵심 증거 시각화 */}
                <section className="card visualization-card grid-span-2">
                    <h2>3️⃣ 핵심 증거 시각화: 편향 이동도</h2>
                    <div className="chart-placeholder">
                        <p></p>
                        <span>(이곳에 Chart.js 또는 D3.js 기반 스캐터 플롯이 렌더링됩니다.)</span>
                        <ul className="chart-legend">
                            <li>🔵 시청 기록</li>
                            <li>🔴 추천 영상</li>
                            <li className="line-legend">--- (y=x 기준선)</li>
                        </ul>
                    </div>
                </section>

                {/* 4. 필터 버블 수준 평가 */}
                <section className="card filter-bubble-card">
                    <h2>4️⃣ 필터 버블 수준 평가</h2>
                    <div className="bubble-meter">
                        {renderBubbleMeter(userData.filterBubble.level)}
                    </div>
                    <div className="bubble-level-text">
                        <strong>4단계: 강함</strong>
                    </div>
                    <ul className="criteria-list">
                        {userData.filterBubble.criteria.map(c => (
                            <li key={c.name} className={c.met ? 'met' : ''}>
                                {c.met ? '✓' : '✗'} {c.name}: <strong>{c.value}</strong>
                            </li>
                        ))}
                    </ul>
                </section>

                {/* 5. 추천 알고리즘 편향 해석 */}
                <section className="card interpretation-card">
                    <h2>5️⃣ LLM 기반 편향 해석</h2>
                    <p>
                        User_Alpha님은 '시장 경제', '정부 정책 비판' 관련 영상을 주로 시청하셨습니다(편향 +{userData.historyBias}). 
                        유튜브의 강화학습 알고리즘은 **'시청 시간 극대화'**를 목표로 합니다.
                    </p>
                    <p>
                        사용자가 기존 성향의 영상에 높은 참여도를 보이자, 알고리즘은 이와 유사하거나 
                        <strong>'더 강한' 성향(+{userData.recoBias})의 영상을 추천</strong>하여 만족도를 높이려 시도합니다. 
                        이 과정이 반복되며 **'되먹임 고리(Feedback Loop)'**가 형성되었습니다.
                    </p>
                </section>

                {/* 6. Exploration Box (상호작용) */}
                <section className="card exploration-card">
                    <h2>6️⃣ 알고리즘 탐험하기</h2>
                    <div className="action-box">
                        <h3>What-if 시뮬레이션</h3>
                        <p>"만약 당신이 <strong>진보적(-0.8) 영상을 10개</strong> 보면, 추천은 어떻게 바뀔까요?"</p>
                        <button onClick={handleSimulateClick}>예측 결과 보기</button>
                    </div>
                    <div className="action-box">
                        <h3>균형 잡힌 콘텐츠 제안</h3>
                        <p>주제 'AI 윤리'에 대한 다양한 시각입니다.</p>
                        <ul className="balanced-list">
                            <li><a href="#" target="_blank">[진보] AI가 초래하는 사회적 불평등</a></li>
                            <li><a href="#" target="_blank">[중립] AI 기술의 현재와 미래 (데이터)</a></li>
                            <li><a href="#" target="_blank">[보수] AI 산업 발전을 위한 규제 완화</a></li>
                        </ul>
                    </div>
                </section>

                {/* 7. Research Report (상호작용) */}
                <section className="card report-card grid-span-2">
                    <h2>7️⃣ 연구 보고서 자동 생성 (요약)</h2>
                    <details className="report-accordion" open={isReportOpen} onToggle={(e) => setIsReportOpen(e.currentTarget.open)}>
                        <summary>
                            {isReportOpen ? '연구 방법론 닫기' : '연구 방법론 상세보기'}
                        </summary>
                        <div className="report-content">
                            <h4>초록 (Abstract)</h4>
                            <p>본 연구는 유튜브 추천 알고리즘이 사용자의 확증 편향에 미치는 영향을 탐지하는 시스템 개발을 목표로 한다...</p>
                            
                            <h4>KoBERT 분류 모델</h4>
                            <p>국내 정치/시사 뉴스 데이터 2만 건을 '진보', '중립', '보수'로 라벨링하여 `kcbert-base` 모델을 파인튜닝(Fine-tuning)하였다. (검증 데이터셋 기준 정확도: 91.2%)</p>

                            <h4>데이터 처리 과정</h4>
                            <pre>
1. YouTube API (History, Recommendation) 데이터 수집
2. 제목/설명/태그 텍스트 전처리
3. KoBERT 모델 입력 → Bias Score (-1.0 ~ +1.0) 출력
4. Bias_History (평균) vs Bias_Rec (평균) 비교
5. Bias Drift 계산 및 필터 버블 지표화
                            </pre>

                            <h4>고찰 및 한계점</h4>
                            <p>본 연구의 모델은 메타데이터(제목, 설명)에 의존하므로, 영상 본문의 시각적/청각적 편향을 직접 측정하지 못하는 한계가 있다...</p>
                        </div>
                    </details>
                </section>
            </main>
        </div>
    );
}

export default Dashboard;