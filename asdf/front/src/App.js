import React, { useState, useEffect } from 'react';
import Hyperspeed from './Hyperspeed';
import { hyperSpeedOptions } from './HyperspeedPresets';
import Dashboard from './Dashboard';

function App() {
    // 'intro' | 'loading' | 'result'
    const [phase, setPhase] = useState('intro');
    const [loadingText, setLoadingText] = useState('Initializing Analysis...');
    const [videoData, setVideoData] = useState(null);

    // 분석 시뮬레이션 시작 함수
    const startAnalysisSequence = (data) => {
        setVideoData(data);
        setPhase('loading');

        const sequence = [
            { text: `Analyzing Watch Logs...`, delay: 0 },
            { text: 'Extracting Keywords & Topics...', delay: 1500 },
            { text: 'Running KoElectra Logic...', delay: 3000 },
            { text: 'Calculating Political Bias Vectors...', delay: 4500 },
            { text: 'Finalizing Report...', delay: 6000 },
        ];

        sequence.forEach(({ text, delay }) => {
            setTimeout(() => setLoadingText(text), delay);
        });

        setTimeout(() => {
            setPhase('result');
        }, 7500);
    };

    useEffect(() => {
        // 1. 저장된 데이터 확인 (새로고침 시)
        const savedData = localStorage.getItem("youtube_data_ids");
        if (savedData) {
            try {
                const parsed = JSON.parse(savedData);
                setVideoData(parsed);
                // 이미 데이터가 있으면 바로 결과로 갈지, 인트로에 남을지 결정.
                // UX상, 새로고침했는데 또 인트로면 귀찮으므로 데이터 있으면 바로 결과창.
                // 단, 처음엔 인트로여야 함.
                // 여기서는 사용자의 요청에 따라 "Intro 유지"로 변경. 
                // 수동으로 넘어갈 수 있는 버튼을 추가했음.

            } catch (e) {
                console.error("데이터 파싱 실패", e);
            }
        }

        // 2. 익스텐션 이벤트 수신
        const handleDataReady = () => {
            console.log("[React] 데이터 도착 이벤트 감지!");
            const newData = localStorage.getItem("youtube_data_ids");
            if (newData) {
                try {
                    const parsed = JSON.parse(newData);
                    // 이벤트로 들어온건 "새로운 분석"이므로 로딩 시퀀스 태움
                    startAnalysisSequence(parsed);
                } catch (e) { console.error(e); }
            }
        };

        window.addEventListener("YoutubeDataReady", handleDataReady);
        return () => window.removeEventListener("YoutubeDataReady", handleDataReady);
    }, []);


    const handleOpenYoutube = () => {
        window.open('https://www.youtube.com/feed/history', '_blank');
    };

    // --- RENDER ---
    return (
        <div className="relative w-screen h-screen overflow-hidden bg-[#09090b] text-white font-sans selection:bg-purple-500/30">

            {/* 1. Background: Hyperspeed (Visible in Intro & Loading) */}
            {phase !== 'result' && (
                <div className="absolute inset-0 z-0">
                    <Hyperspeed
                        loading={phase === 'loading'}
                        effectOptions={hyperSpeedOptions}
                    />
                    {/* Dark Overlay for better text readability in Intro */}
                    <div className={`absolute inset-0 bg-black/40 transition-opacity duration-1000 ${phase === 'loading' ? 'opacity-80' : 'opacity-40'}`} />
                </div>
            )}

            {/* 2. Phase: Intro */}
            {phase === 'intro' && (
                <div className="relative z-10 flex flex-col items-center justify-center min-h-screen p-6 animate-in fade-in duration-700 overflow-y-auto custom-scrollbar">

                    <div className="max-w-4xl w-full z-10">
                        <div className="text-center mb-16 space-y-4">
                            <h1 className="text-5xl md:text-7xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-zinc-400 drop-shadow-2xl">
                                ALGO<span className="text-purple-500">.</span>RITHM
                            </h1>
                            <p className="text-xl text-zinc-300 max-w-2xl mx-auto leading-relaxed drop-shadow-lg font-medium">
                                Unlock your YouTube bias. <br className="hidden md:block" />
                                Follow the 3-step synchronization process below.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
                            {/* Step 1 */}
                            <div className="group relative bg-black/40 border border-white/10 rounded-3xl p-8 hover:bg-black/60 hover:border-white/20 transition-all duration-300 hover:-translate-y-1 backdrop-blur-md">
                                <div className="absolute -top-4 -left-4 w-12 h-12 bg-zinc-900 border border-zinc-700 rounded-2xl flex items-center justify-center font-bold text-xl text-purple-400 shadow-xl">1</div>
                                <div className="h-12 w-12 bg-red-500/10 rounded-xl flex items-center justify-center mb-6 group-hover:bg-red-500/20 transition-colors">
                                    <svg className="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 24 24"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" /></svg>
                                </div>
                                <h3 className="text-xl font-bold mb-2">Open History</h3>
                                <p className="text-zinc-400 text-sm mb-6 h-10">Access your YouTube watch history page.</p>
                                <button onClick={handleOpenYoutube} className="w-full py-3 bg-red-600 hover:bg-red-700 text-white font-bold rounded-xl transition-colors flex items-center justify-center gap-2 group-last:hidden shadow-lg">
                                    Go to Youtube
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                                </button>
                            </div>

                            {/* Step 2 */}
                            <div className="group relative bg-black/40 border border-white/10 rounded-3xl p-8 hover:bg-black/60 hover:border-white/20 transition-all duration-300 hover:-translate-y-1 backdrop-blur-md">
                                <div className="absolute -top-4 -left-4 w-12 h-12 bg-zinc-900 border border-zinc-700 rounded-2xl flex items-center justify-center font-bold text-xl text-purple-400 shadow-xl">2</div>
                                <div className="h-12 w-12 bg-purple-500/10 rounded-xl flex items-center justify-center mb-6 group-hover:bg-purple-500/20 transition-colors">
                                    <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                                </div>
                                <h3 className="text-xl font-bold mb-2">Click Extension</h3>
                                <p className="text-zinc-400 text-sm mb-4">Click the puzzle icon <span className="inline-block bg-zinc-700 px-1 rounded">🧩</span> in Chrome, then select <strong>'Algo.Rithm'</strong>.</p>
                                <div className="w-full h-10 rounded-lg bg-zinc-800/80 flex items-center justify-center text-xs text-zinc-500 font-mono border border-white/5">
                                    Top-right of browser
                                </div>
                            </div>

                            {/* Step 3 */}
                            <div className="group relative bg-black/40 border border-white/10 rounded-3xl p-8 hover:bg-black/60 hover:border-white/20 transition-all duration-300 hover:-translate-y-1 backdrop-blur-md">
                                <div className="absolute -top-4 -left-4 w-12 h-12 bg-zinc-900 border border-zinc-700 rounded-2xl flex items-center justify-center font-bold text-xl text-cyan-400 shadow-xl">3</div>
                                <div className="h-12 w-12 bg-cyan-500/10 rounded-xl flex items-center justify-center mb-6 group-hover:bg-cyan-500/20 transition-colors">
                                    <svg className="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                                </div>
                                <h3 className="text-xl font-bold mb-2">Run Analysis</h3>
                                <p className="text-zinc-400 text-sm mb-4">Click the <strong>'START DIAGNOSIS'</strong> button in the popup.</p>
                                <div className="w-full h-10 rounded-lg bg-gradient-to-r from-purple-500/20 to-cyan-500/20 flex items-center justify-center text-xs text-purple-300 font-bold border border-purple-500/20 animate-pulse shadow-[0_0_15px_rgba(168,85,247,0.3)]">
                                    Waiting for click...
                                </div>
                            </div>
                        </div>

                        {/* Waiting Indicator */}
                        <div className="flex flex-col items-center justify-center gap-3">
                            <div className="relative flex h-3 w-3">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500"></span>
                            </div>
                            <p className="text-sm font-mono text-zinc-400 tracking-widest text-center drop-shadow-md">
                                LISTENING FOR EXTENSION SIGNAL...
                            </p>
                        </div>

                        {/* Resume Button */}
                        {videoData && (
                            <div className="mt-8 text-center animate-in fade-in slide-in-from-bottom-4 duration-1000">
                                <button
                                    onClick={() => setPhase('result')}
                                    className="px-6 py-2 rounded-full border border-white/10 bg-black/40 hover:bg-black/60 text-zinc-300 text-sm font-mono transition-all flex items-center mx-auto gap-2 group backdrop-blur-md"
                                >
                                    <span>Recover Previous Session</span>
                                    <span className="group-hover:translate-x-1 transition-transform">→</span>
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* 3. Phase: Loading */}
            {phase === 'loading' && (
                <div className="relative z-10 flex flex-col items-center justify-center h-full pointer-events-none animate-in fade-in duration-1000">
                    <h1 className="text-4xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 animate-pulse text-center drop-shadow-2xl">
                        HYPER PROCESSING
                    </h1>
                    <div className="mt-12 flex flex-col items-center gap-6">
                        <div className="w-80 h-1 bg-white/10 rounded-full overflow-hidden backdrop-blur-sm border border-white/5">
                            <div className="h-full bg-gradient-to-r from-purple-500 to-cyan-400 w-1/2 animate-[shimmer_1.5s_infinite_ease-in-out]"></div>
                        </div>
                        <p className="text-xl text-white/90 font-mono tracking-widest uppercase drop-shadow-lg">
                            {loadingText}
                        </p>
                    </div>
                </div>
            )}

            {/* 4. Phase: Result */}
            {phase === 'result' && (
                <div className="relative z-20 w-full h-full bg-[#09090b] overflow-y-auto animate-in fade-in zoom-in-95 duration-700">
                    <Dashboard rawLabels={videoData} />
                </div>
            )}
        </div>
    );
}

export default App;