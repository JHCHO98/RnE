import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, CircleDashed, Chrome, Youtube, BrainCircuit, Sparkles } from 'lucide-react';

const steps = [
    { id: 1, label: 'Accessing Chrome History', icon: Chrome, desc: 'Syncing search patterns...' },
    { id: 2, label: 'YouTube API Gateway', icon: Youtube, desc: 'Fetching watch logs...' },
    { id: 3, label: 'KoElectra Inference', icon: BrainCircuit, desc: 'Detecting political bias...' },
    { id: 4, label: 'Gemini 3.0 Processing', icon: Sparkles, desc: 'Generating final report...' },
];

const AnalysisSteps = ({ currentStep }) => {
    return (
        <div className="w-full max-w-md bg-black/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl">
            <div className="space-y-6">
                <div className="text-center mb-8">
                    <h2 className="text-2xl font-bold text-white tracking-tight">System Analysis</h2>
                    <p className="text-zinc-400 text-sm">Processing hyperspeed data stream</p>
                </div>

                <div className="relative space-y-6">
                    {/* Connecting Line */}
                    <div className="absolute left-6 top-2 bottom-2 w-0.5 bg-zinc-800 -z-10" />

                    {steps.map((step, index) => {
                        const Icon = step.icon;
                        const isActive = currentStep === index + 1;
                        const isCompleted = currentStep > index + 1;

                        return (
                            <motion.div
                                key={step.id}
                                initial={{ opacity: 0.5, x: -10 }}
                                animate={{
                                    opacity: isActive || isCompleted ? 1 : 0.4,
                                    x: 0,
                                    scale: isActive ? 1.05 : 1
                                }}
                                className={`flex items-center gap-4 relative ${isActive ? 'bg-white/5 rounded-xl -m-2 p-2 border border-white/5' : 'p-2'}`}
                            >
                                <div className={`
                                    relative flex items-center justify-center w-8 h-8 rounded-full border-2 
                                    transition-all duration-500
                                    ${isCompleted ? 'bg-green-500 border-green-500 text-black' :
                                        isActive ? 'bg-purple-500 border-purple-500 text-white shadow-[0_0_15px_rgba(168,85,247,0.5)]' :
                                            'bg-black border-zinc-700 text-zinc-700'}
                                `}>
                                    {isCompleted ? <CheckCircle2 size={16} /> :
                                        isActive ? <CircleDashed size={16} className="animate-spin" /> :
                                            <div className="w-2 h-2 rounded-full bg-zinc-700" />}
                                </div>

                                <div className="flex-1">
                                    <div className="flex items-center justify-between">
                                        <h3 className={`font-bold text-sm ${isActive || isCompleted ? 'text-white' : 'text-zinc-500'}`}>
                                            {step.label}
                                        </h3>
                                        <Icon size={16} className={`${isActive ? 'text-purple-400' : 'text-zinc-600'}`} />
                                    </div>
                                    {isActive && (
                                        <motion.p
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            className="text-xs text-purple-300 mt-1 font-mono"
                                        >
                                            {step.desc}
                                        </motion.p>
                                    )}
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Progress Bar */}
            <div className="mt-8 h-1 w-full bg-zinc-800 rounded-full overflow-hidden">
                <motion.div
                    className="h-full bg-gradient-to-r from-purple-500 to-indigo-500"
                    initial={{ width: "0%" }}
                    animate={{ width: `${(currentStep / 5) * 100}%` }}
                    transition={{ duration: 0.5 }}
                />
            </div>
        </div>
    );
};

export default AnalysisSteps;
