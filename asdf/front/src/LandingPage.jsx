import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, BrainCircuit, Activity, Database } from 'lucide-react';
import Hyperspeed from './Hyperspeed'; // 사용자가 가지고 있는 컴포넌트 경로
import { hyperSpeedOptions } from './HyperspeedPresets'; // 위에서 만든 설정

const LandingPage = ({ onStart }) => {
    const [isSpeeding, setIsSpeeding] = useState(false);

    // Hyperspeed 효과를 제어하기 위한 ref (라이브러리 구현 방식에 따라 다를 수 있음)
    // 여기서는 state 변경 시 Hyperspeed 컴포넌트가 props로 반응한다고 가정하거나,
    // 시각적으로 UI만 처리합니다.

    const handleStartClick = () => {
        setIsSpeeding(true);
        // 1.5초 동안 워프 효과를 보여준 뒤 다음 페이지로 이동
        setTimeout(() => {
            onStart();
        }, 1500);
    };

    return (
        <div className="relative min-h-screen bg-[#09090b] text-white overflow-hidden font-sans">

            {/* --- [BACKGROUND] Hyperspeed Component --- */}
            <div className="absolute inset-0 z-0 opacity-80">
                <Hyperspeed
                    effectOptions={{
                        ...hyperSpeedOptions,
                        // 버튼을 눌렀을 때 speedUp 상태가 되도록 라이브러리가 지원한다면 여기서 제어
                        // 지원하지 않는다면 단순 배경으로 작동
                    }}
                />
                {/* 가속 시 화면이 하얗게 변하는 워프 효과 오버레이 */}
                <motion.div
                    animate={{ opacity: isSpeeding ? 1 : 0 }}
                    transition={{ duration: 1 }}
                    className="absolute inset-0 bg-white pointer-events-none mix-blend-overlay"
                />
            </div>

            {/* --- [CONTENT] --- */}
            {/* 배경 클릭을 방해하지 않으려면 pointer-events-none 사용, 버튼만 auto로 풂 */}
            <div className={`relative z-10 flex flex-col h-screen transition-opacity duration-1000 ${isSpeeding ? 'opacity-0' : 'opacity-100'}`}>

                {/* Navbar */}
                <nav className="w-full max-w-7xl mx-auto px-6 py-6 flex justify-between items-center">
                    <div className="font-bold text-xl tracking-tighter flex items-center gap-2">
                        <div className="w-8 h-8 bg-white/10 backdrop-blur-md border border-white/20 rounded-lg flex items-center justify-center">
                            <BrainCircuit size={18} className="text-white" />
                        </div>
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-white/60">Algo.Analysis</span>
                    </div>
                </nav>

                {/* Hero Section */}
                <main className="flex-1 flex flex-col items-center justify-center text-center px-4">
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 1, delay: 0.2 }}
                        className="space-y-8 max-w-5xl"
                    >
                        {/* Tagline */}
                        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-purple-500/30 bg-black/40 backdrop-blur-md text-purple-300 text-xs font-medium">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
                            </span>
                            Real-time Bias Monitoring
                        </div>

                        {/* Main Title */}
                        <h1 className="text-6xl md:text-8xl font-black tracking-tighter leading-[0.9] text-white drop-shadow-2xl">
                            ARE YOU <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-indigo-400 to-emerald-400">
                                DRIVING?
                            </span>
                        </h1>

                        <p className="text-xl md:text-2xl text-zinc-300 max-w-2xl mx-auto font-light leading-relaxed drop-shadow-md">
                            우리는 매일 알고리즘이라는 <strong>무한한 고속도로</strong>를 질주합니다.<br />
                            당신의 핸들은 지금 어디로 꺾여 있나요?
                        </p>

                        {/* CTA Button */}
                        <div className="pt-10">
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={handleStartClick}
                                className="group relative inline-flex items-center gap-4 px-10 py-5 bg-white text-black rounded-full font-bold text-xl shadow-[0_0_40px_rgba(255,255,255,0.4)] hover:shadow-[0_0_60px_rgba(139,92,246,0.6)] transition-all overflow-hidden"
                            >
                                <span className="relative z-10">알고리즘 진단 시작</span>
                                <ArrowRight className="relative z-10 group-hover:translate-x-1 transition-transform" />

                                {/* Button Hover Gradient */}
                                <div className="absolute inset-0 bg-gradient-to-r from-purple-200 to-indigo-200 opacity-0 group-hover:opacity-100 transition-opacity" />
                            </motion.button>

                            <p className="mt-4 text-xs text-zinc-500 font-mono">
                                * Click to accelerate data collection
                            </p>
                        </div>
                    </motion.div>
                </main>

                {/* Footer Features (Glassmorphism) */}
                <div className="w-full max-w-7xl mx-auto px-6 pb-12 grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
                    <GlassFeature icon={Activity} title="Infinite Scroll" desc="끝없는 피드 속 확증 편향 패턴 분석" />
                    <GlassFeature icon={Database} title="Data Velocity" desc="초고속으로 수집되는 시청 기록 처리" />
                    <GlassFeature icon={BrainCircuit} title="Steering AI" desc="KoElectra & Gemini가 방향성을 진단" />
                </div>
            </div>
        </div>
    );
};

// 작은 카드 컴포넌트
const GlassFeature = ({ icon: Icon, title, desc }) => (
    <div className="p-4 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md flex items-center gap-4 hover:bg-white/5 transition-colors">
        <div className="p-2 rounded-lg bg-white/10 text-purple-300">
            <Icon size={20} />
        </div>
        <div>
            <h3 className="font-bold text-white text-sm">{title}</h3>
            <p className="text-xs text-zinc-400">{desc}</p>
        </div>
    </div>
);

export default LandingPage;