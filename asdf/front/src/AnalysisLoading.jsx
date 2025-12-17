import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Loader2, CheckCircle2 } from 'lucide-react';

const AnalysisLoading = ({ onComplete }) => {
    const [step, setStep] = useState(0);

    const steps = [
        { text: "유튜브 시청 기록 수집 중...", sub: "최근 6개월 데이터 스크래핑" },
        { text: "KoElectra 자연어 처리 분석 중...", sub: "영상 제목 및 자막 토큰화 진행" },
        { text: "Gemini 3.0 편향성 진단 중...", sub: "벡터 임베딩 및 유사도 계산" },
        { text: "리포트 생성 완료!", sub: "대시보드로 이동합니다..." }
    ];

    // 단계별 텍스트 자동 전환 로직 (각 단계 2초)
    useEffect(() => {
        if (step < steps.length - 1) {
            const timer = setTimeout(() => setStep(prev => prev + 1), 2000);
            return () => clearTimeout(timer);
        } else {
            // 마지막 단계에서 부모 컴포넌트에 완료 신호 전송
            const timer = setTimeout(() => onComplete && onComplete(), 1000);
            return () => clearTimeout(timer);
        }
    }, [step, steps.length, onComplete]);

    return (
        <div className="min-h-screen bg-[#09090b] flex flex-col items-center justify-center relative overflow-hidden text-white font-sans">

            {/* Background Grid Pattern */}
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />

            {/* Central Radar Animation */}
            <div className="relative flex items-center justify-center mb-12">
                {/* Pulsing Circles */}
                {[0, 1, 2].map((i) => (
                    <motion.div
                        key={i}
                        className="absolute border border-purple-500/30 rounded-full"
                        initial={{ width: '100px', height: '100px', opacity: 0.8 }}
                        animate={{
                            width: ['100px', '400px'],
                            height: ['100px', '400px'],
                            opacity: [0.5, 0]
                        }}
                        transition={{
                            duration: 2,
                            repeat: Infinity,
                            delay: i * 0.6,
                            ease: "easeOut"
                        }}
                    />
                ))}

                {/* Center Core */}
                <div className="w-24 h-24 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-full flex items-center justify-center shadow-[0_0_50px_rgba(124,58,237,0.5)] z-10 relative">
                    <Loader2 className="animate-spin text-white/80" size={40} />
                    <div className="absolute inset-0 rounded-full border-t-2 border-white/30 animate-spin" style={{ animationDuration: '3s' }} />
                </div>
            </div>

            {/* Dynamic Text Section */}
            <div className="text-center z-10 h-24">
                <motion.div
                    key={step} // step이 바뀔 때마다 애니메이션 재실행
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-2"
                >
                    <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">
                        {steps[step].text}
                    </h2>
                    <p className="text-sm text-zinc-500 font-mono">
                        {steps[step].sub}
                    </p>
                </motion.div>
            </div>

            {/* Step Indicators */}
            <div className="flex gap-2 mt-8 z-10">
                {steps.map((_, i) => (
                    <div
                        key={i}
                        className={`h-1.5 rounded-full transition-all duration-500 ${i <= step ? 'w-8 bg-purple-500' : 'w-2 bg-zinc-800'
                            }`}
                    />
                ))}
            </div>

        </div>
    );
};

export default AnalysisLoading;