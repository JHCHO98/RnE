import React from 'react';
import { ExternalLink, AlertTriangle, ShieldCheck, Activity } from 'lucide-react';

const ExtensionPopup = () => {
    // Mock Data: 현재 감지된 영상 상태 (Safe / Caution / Danger)
    const currentVideoStatus = 'danger'; // 테스트용
    const biasScore = 0.85;

    return (
        <div className="w-[350px] h-[500px] bg-[#09090b] text-white font-sans flex flex-col overflow-hidden border border-zinc-800 shadow-2xl relative">
            {/* Background Glow */}
            <div className="absolute top-[-50px] left-[-50px] w-32 h-32 bg-purple-600/20 rounded-full blur-[50px]" />

            {/* Header */}
            <header className="flex items-center justify-between p-5 z-10 border-b border-zinc-800/50 bg-[#09090b]/80 backdrop-blur-md">
                <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-purple-600 rounded-md flex items-center justify-center">
                        <Activity size={14} className="text-white" />
                    </div>
                    <span className="font-bold text-sm tracking-wide">Algo.Guard</span>
                </div>
                <div className="text-[10px] bg-zinc-800 px-2 py-1 rounded-full text-zinc-400">v1.0.2</div>
            </header>

            {/* Main Content */}
            <main className="flex-1 p-5 flex flex-col gap-6 z-10">

                {/* 1. Real-time Status Card */}
                <div className={`
          relative p-5 rounded-2xl border backdrop-blur-sm transition-all
          ${currentVideoStatus === 'danger'
                        ? 'bg-red-500/10 border-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.15)]'
                        : 'bg-emerald-500/10 border-emerald-500/30'}
        `}>
                    <div className="flex items-start gap-3">
                        {currentVideoStatus === 'danger' ? (
                            <AlertTriangle className="text-red-500 shrink-0 animate-pulse" size={24} />
                        ) : (
                            <ShieldCheck className="text-emerald-500 shrink-0" size={24} />
                        )}
                        <div>
                            <h3 className={`font-bold text-lg ${currentVideoStatus === 'danger' ? 'text-red-400' : 'text-emerald-400'}`}>
                                {currentVideoStatus === 'danger' ? '편향 위험 감지' : '안전한 콘텐츠'}
                            </h3>
                            <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
                                현재 시청 중인 영상은
                                <strong className="text-zinc-200 mx-1">'극우'</strong>
                                성향이 매우 강합니다. (편향도 {biasScore})
                            </p>
                        </div>
                    </div>

                    {/* Progress Bar inside Card */}
                    <div className="mt-4">
                        <div className="flex justify-between text-[10px] text-zinc-500 mb-1 uppercase font-mono">
                            <span>Neutral</span>
                            <span>Extreme</span>
                        </div>
                        <div className="h-1.5 w-full bg-zinc-900 rounded-full overflow-hidden">
                            <div
                                className={`h-full rounded-full ${currentVideoStatus === 'danger' ? 'bg-red-500' : 'bg-emerald-500'}`}
                                style={{ width: `${biasScore * 100}%` }}
                            />
                        </div>
                    </div>
                </div>

                {/* 2. My Status Summary */}
                <div className="space-y-3">
                    <h4 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Today's Summary</h4>
                    <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl flex items-center justify-between">
                        <div>
                            <div className="text-2xl font-mono font-bold text-white">42<span className="text-sm text-zinc-500 font-sans ml-1">건</span></div>
                            <div className="text-[10px] text-zinc-500">분석된 영상</div>
                        </div>
                        <div className="h-8 w-[1px] bg-zinc-800" />
                        <div className="text-right">
                            <div className="text-2xl font-mono font-bold text-purple-400">+0.4</div>
                            <div className="text-[10px] text-zinc-500">누적 편향 점수</div>
                        </div>
                    </div>
                </div>
            </main>

            {/* Footer CTA */}
            <footer className="p-5 border-t border-zinc-800/50 bg-[#09090b]">
                <button
                    onClick={() => window.open('http://localhost:3000', '_blank')}
                    className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3.5 rounded-xl transition-all shadow-lg hover:shadow-indigo-500/25 active:scale-95"
                >
                    <span>상세 분석 리포트 확인</span>
                    <ExternalLink size={16} />
                </button>
            </footer>
        </div>
    );
};

export default ExtensionPopup;