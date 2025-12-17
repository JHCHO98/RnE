import React, { useState, useEffect } from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip as RechartsTooltip, CartesianGrid, ReferenceArea, ReferenceLine,
  BarChart, Bar, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ComposedChart, Line, Area
} from 'recharts';
import {
  LayoutDashboard, Activity, BrainCircuit, Search, ChevronRight, Info,
  MousePointer2, Network, ShieldAlert, List, PieChart as PieIcon,
  AlertOctagon, Lock, MapPin, Tag
} from 'lucide-react';
import { motion, animate } from 'framer-motion';
import { CloudLightning as ListItem } from 'lucide-react';

/**
 * ============================================================================
 * [ANIMATION COMPONENTS]
 * ============================================================================
 */
const Counter = ({ value, duration = 2 }) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const controls = animate(0, value, {
      duration,
      onUpdate: (v) => setDisplayValue(Math.floor(v)),
      ease: "easeOut"
    });
    return controls.stop;
  }, [value, duration]);

  return <span>{displayValue.toLocaleString()}</span>;
};

/**
 * ============================================================================
 * [DATA] 데이터셋
 * ============================================================================
 */
// 1. 상세 주제 분류 데이터 (Page 1 Filling)
const DETAILED_INTERESTS = [
  { name: 'Politics & Society', value: 35.5, color: '#ef4444' },
  { name: 'Gaming (FPS)', value: 12.0, color: '#3b82f6' },
  { name: 'Animals / Pets', value: 8.5, color: '#60a5fa' },
  { name: 'Science & Tech', value: 15.0, color: '#10b981' },
  { name: 'Economics', value: 10.2, color: '#f59e0b' },
  { name: 'Entertainment', value: 8.8, color: '#8b5cf6' },
  { name: 'Music', value: 5.0, color: '#d946ef' },
  { name: 'Vlog / Lifestyle', value: 3.0, color: '#f472b6' },
  { name: 'Education', value: 2.0, color: '#64748b' },
];

// 2. 정치 성향 벡터 맵 (Page 2)
// 2. 정치 성향 변화 추이 (Bias Drift) - Time Series Analysis
// sequence: 시청 순서
// bias: 개별 영상의 성향 점수 (-1: 진보, +1: 보수)
const BIAS_DRIFT_DATA = [
  // 1~10: 초기 탐색 (중립 ~ 약한 편향)
  { sequence: 1, bias: 0.02, title: "Daily Vlog: Morning Routine" },
  { sequence: 2, bias: -0.15, title: "Tech Review: New Smartphone" },
  { sequence: 3, bias: 0.10, title: "News: Economy Update" },
  { sequence: 4, bias: 0.05, title: "Travel: Visiting Seoul" },
  { sequence: 5, bias: 0.25, title: "Opinion: Current Policy" },
  { sequence: 6, bias: -0.05, title: "Food: Best Burger Spots" },
  { sequence: 7, bias: 0.30, title: "Debate: Tax Issues" },
  { sequence: 8, bias: 0.15, title: "History: Modern Korea" },
  { sequence: 9, bias: 0.35, title: "Analysis: Election Trends" },
  { sequence: 10, bias: 0.20, title: "Documentary: Nature" },

  // 11~30: 알고리즘 추천 시작 (약보수 진입)
  { sequence: 11, bias: 0.45, title: "Why Policy A is Failing" },
  { sequence: 12, bias: 0.38, title: "Reaction: News Highlights" },
  { sequence: 13, bias: 0.50, title: "Hidden Truth about X" },
  { sequence: 14, bias: 0.42, title: "Economy Forecast 2024" },
  { sequence: 15, bias: 0.10, title: "Sports: Soccer Highlights" },
  { sequence: 16, bias: 0.55, title: "Criticism: Opposition Party" },
  { sequence: 17, bias: 0.48, title: "Commentary: Social Issues" },
  { sequence: 18, bias: 0.60, title: "Interview with Expert A" },
  { sequence: 19, bias: 0.25, title: "Movie Review: Action" },
  { sequence: 20, bias: 0.65, title: "Focus: National Security" },
  { sequence: 21, bias: 0.58, title: "The Real Problem is..." },
  { sequence: 22, bias: 0.70, title: "Shocking Fact Reveal" },
  { sequence: 23, bias: 0.62, title: "Review: Political Book" },
  { sequence: 24, bias: 0.35, title: "Gaming: Live Stream" },
  { sequence: 25, bias: 0.72, title: "Why Mainstream Media Lies" },
  { sequence: 26, bias: 0.68, title: "Defense Strategy Analysis" },
  { sequence: 27, bias: 0.75, title: "Must Watch: The Truth" },
  { sequence: 28, bias: 0.40, title: "Global News Summary" },
  { sequence: 29, bias: 0.78, title: "Urgent: Please Share" },
  { sequence: 30, bias: 0.80, title: "Live: Protest Scene" },

  // 31~60: 필터버블 형성 (강보수 고착화)
  { sequence: 31, bias: 0.82, title: "They Don't Want You to Know" },
  { sequence: 32, bias: 0.75, title: "Analysis: Candidate B" },
  { sequence: 33, bias: 0.85, title: "Full Breakdown of Issue" },
  { sequence: 34, bias: 0.79, title: "Comment: Recent Scandal" },
  { sequence: 35, bias: 0.88, title: "Exclusive Coverage" },
  { sequence: 36, bias: 0.81, title: "Fact Check: The Lies" },
  { sequence: 37, bias: 0.90, title: "Warning to Citizens" },
  { sequence: 38, bias: 0.84, title: "Deep Dive: Corruption" },
  { sequence: 39, bias: 0.86, title: "Voice of the People" },
  { sequence: 40, bias: 0.92, title: "Critical Alert!" },
  { sequence: 41, bias: 0.83, title: "Review: Biased News" },
  { sequence: 42, bias: 0.89, title: "Unfiltered Truth" },
  { sequence: 43, bias: 0.85, title: "Debunking the Myths" },
  { sequence: 44, bias: 0.91, title: "Patriot's Perspective" },
  { sequence: 45, bias: 0.87, title: "Hidden Agenda Exposed" },
  { sequence: 46, bias: 0.93, title: "Final Warning" },
  { sequence: 47, bias: 0.88, title: "Special Report: Crisis" },
  { sequence: 48, bias: 0.90, title: "Who is Behind This?" },
  { sequence: 49, bias: 0.82, title: "Summary of Events" },
  { sequence: 50, bias: 0.94, title: "Truth Bomb dropped" },
  { sequence: 51, bias: 0.86, title: "Don't be Fooled" },
  { sequence: 52, bias: 0.89, title: "System Collapse?" },
  { sequence: 53, bias: 0.91, title: "Real Talk: Politics" },
  { sequence: 54, bias: 0.85, title: "Media Silence" },
  { sequence: 55, bias: 0.95, title: "Absolute Proof" },
  { sequence: 56, bias: 0.88, title: "Wake Up Call" },
  { sequence: 57, bias: 0.92, title: "Secret Documents" },
  { sequence: 58, bias: 0.84, title: "Comparison: Then vs Now" },
  { sequence: 59, bias: 0.90, title: "The End of Liberty?" },
  { sequence: 60, bias: 0.96, title: "Viral Commentary" },

  // 61~100: 확증편향 심화 (극단적 쏠림 및 가끔 중립 클릭 시도)
  { sequence: 61, bias: 0.89, title: "Resist the Agenda" },
  { sequence: 62, bias: 0.93, title: "Logic vs Emotion" },
  { sequence: 63, bias: 0.91, title: "Destroying the Narrative" },
  { sequence: 64, bias: 0.87, title: "Expert Panel Discussion" },
  { sequence: 65, bias: 0.94, title: "Shocking Betrayal" },
  { sequence: 66, bias: 0.90, title: "History Will Remember" },
  { sequence: 67, bias: -0.10, title: "Cat Video (Ad)" }, // 가끔 튀는 값
  { sequence: 68, bias: 0.92, title: "Back to Reality" },
  { sequence: 69, bias: 0.95, title: "Ultimate Guide" },
  { sequence: 70, bias: 0.88, title: "Question Everything" },
  { sequence: 71, bias: 0.91, title: "Silence is Consent" },
  { sequence: 72, bias: 0.96, title: "Red Pill Moment" },
  { sequence: 73, bias: 0.89, title: "Counter Argument" },
  { sequence: 74, bias: 0.93, title: "Proof in the Pudding" },
  { sequence: 75, bias: 0.85, title: "Weekly Roundup" },
  { sequence: 76, bias: 0.94, title: "They Panicked!" },
  { sequence: 77, bias: 0.97, title: "Victory is Near" },
  { sequence: 78, bias: 0.90, title: "Stop the Madness" },
  { sequence: 79, bias: 0.92, title: "Common Sense" },
  { sequence: 80, bias: 0.95, title: "Freedom Speech" },
  { sequence: 81, bias: 0.88, title: "Behind the Scenes" },
  { sequence: 82, bias: 0.93, title: "Leaked Audio" },
  { sequence: 83, bias: 0.91, title: "No More Lies" },
  { sequence: 84, bias: 0.96, title: "Hard Truths" },
  { sequence: 85, bias: 0.89, title: "Exposing the System" },
  { sequence: 86, bias: 0.94, title: "Globalist Agenda" },
  { sequence: 87, bias: 0.92, title: "Protect Our Future" },
  { sequence: 88, bias: 0.97, title: "Emergency Broadcast" },
  { sequence: 89, bias: 0.90, title: "Think for Yourself" },
  { sequence: 90, bias: 0.95, title: "The Great Awakening" },
  { sequence: 91, bias: 0.93, title: "Stand Your Ground" },
  { sequence: 92, bias: 0.96, title: "Final Battle" },
  { sequence: 93, bias: 0.91, title: "Corrupt Media" },
  { sequence: 94, bias: 0.94, title: "Truth Prevails" },
  { sequence: 95, bias: 0.98, title: "Must See TV" },
  { sequence: 96, bias: 0.92, title: "Identity Politics" },
  { sequence: 97, bias: 0.95, title: "Culture War Update" },
  { sequence: 98, bias: 0.89, title: "Silent Majority" },
  { sequence: 99, bias: 0.96, title: "Game Over" },
  { sequence: 100, bias: 0.97, title: "Final Summary: Conclusion" }
];

// 3. 혐오/위험도 분석 데이터 (Page 2 Risk)
const RISK_RADAR_DATA = [
  { subject: 'Political Extremism', A: 120, fullMark: 150 },
  { subject: 'Hate Speech', A: 45, fullMark: 150 },
  { subject: 'Echo Chamber', A: 140, fullMark: 150 }, // 확증편향 높음
  { subject: 'Violence', A: 20, fullMark: 150 },
  { subject: 'Misinformation', A: 80, fullMark: 150 },
  { subject: 'Sexual Content', A: 10, fullMark: 150 },
];

/**
 * ============================================================================
 * [COMPONENTS] Shared Components
 * ============================================================================
 */
const BentoCard = ({ title, subtitle, children, className = "", icon: Icon }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
    className={`
      flex flex-col overflow-hidden rounded-3xl
      border border-white/10 bg-black/40 backdrop-blur-xl
      hover:border-white/20 hover:bg-black/50 transition-colors
      ${className}
    `}
  >
    <div className="p-8 h-full flex flex-col relative">
      {/* Header */}
      <div className="flex items-start justify-between mb-8 z-10">
        <div>
          <h3 className="flex items-center gap-3 text-2xl font-bold tracking-tight text-white/90">
            {Icon && <Icon size={24} className="text-purple-400" />}
            {title}
          </h3>
          {subtitle && <p className="mt-2 font-mono text-sm text-zinc-500 tracking-widest uppercase">{subtitle}</p>}
        </div>
      </div>
      {/* Content */}
      <div className="flex-1 relative z-10 min-h-0">{children}</div>

      {/* Decor */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full blur-2xl pointer-events-none" />
    </div>
  </motion.div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-lg border border-white/10 bg-black/90 p-3 shadow-xl backdrop-blur-md">
        <p className="text-xs font-bold text-white mb-1">{label || payload[0].payload.name}</p>
        {payload.map((entry, index) => {
          // Hide Y and Z from tooltip
          if (entry.name === 'y' || entry.name === 'z' || entry.name === 'Intensity') return null;

          // Rename X to Similarity
          let name = entry.name;
          let val = entry.value;
          if (entry.name === 'Left/Right' || entry.name === 'Political Spectrum' || entry.name === 'x') {
            name = 'Similarity';
            val = entry.value > 0 ? `+${entry.value} (Cons)` : `${entry.value} (Lib)`;
          }

          return (
            <div key={index} className="text-xs text-zinc-400">
              {name}: <span className="text-purple-400 font-mono">{val}</span>
            </div>
          )
        })}
      </div>
    );
  }
  return null;
};

/**
 * ============================================================================
 * [VIEWS] 
 * ============================================================================
 */

// --- PAGE 1: User Profile & Topics (Scaled Up) ---
const ProfileView = () => (
  <div className="space-y-8 h-full flex flex-col">
    {/* Top Summary */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      <BentoCard title="Total Watched" icon={Activity} className="h-48">
        <div className="flex items-end gap-3 h-full pb-2">
          <span className="text-6xl font-mono font-bold text-white"><Counter value={1240} /></span>
          <span className="text-zinc-500 mb-3 text-xl">videos</span>
        </div>
      </BentoCard>
      <BentoCard title="Top Category" icon={PieIcon} className="h-48">
        <div className="flex items-center h-full text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-orange-400">
          Politics & Society
        </div>
      </BentoCard>
      <BentoCard title="Avg Daily Time" icon={ListItem} className="h-48">
        <div className="flex items-end gap-3 h-full pb-2">
          <span className="text-6xl font-mono font-bold text-white">4.2</span>
          <span className="text-zinc-500 mb-3 text-xl">hours</span>
        </div>
      </BentoCard>
    </div>

    {/* Middle: Chart */}
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 flex-1 min-h-[500px]">
      <BentoCard title="Interest Distribution" subtitle="MACRO ANALYSIS" className="lg:col-span-2" icon={PieIcon}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={DETAILED_INTERESTS}
              cx="50%" cy="50%"
              innerRadius={120} outerRadius={180}
              paddingAngle={3}
              dataKey="value"
            >
              {DETAILED_INTERESTS.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} stroke="rgba(0,0,0,0.5)" />
              ))}
            </Pie>
            <RechartsTooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
        {/* Center Label */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="text-xl text-zinc-500 mb-2">DOMINANT</div>
            <div className="text-4xl font-bold text-white">Politics</div>
          </div>
        </div>
      </BentoCard>

      {/* Right: Detailed List */}
      <BentoCard title="Breakdown" subtitle="FULL TAXONOMY" className="overflow-y-auto custom-scrollbar" icon={List}>
        <div className="space-y-6 pr-4">
          {DETAILED_INTERESTS.map((item, idx) => (
            <div key={idx} className="group flex items-center justify-between p-4 rounded-2xl bg-white/5 hover:bg-white/10 transition-colors border border-white/5">
              <div className="flex items-center gap-4">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-xl text-zinc-300 font-medium group-hover:text-white transition-colors">
                  {item.name}
                </span>
              </div>
              <span className="font-mono text-xl text-white/50 group-hover:text-purple-400">
                {item.value}%
              </span>
            </div>
          ))}
        </div>
      </BentoCard>
    </div>
  </div>
);

// Need to define ListItem icon



// --- PAGE 2: Bias & Risk Diagnosis (Scaled for Showcase) ---
const BiasView = () => (
  <div className="h-full flex flex-col gap-8">
    {/* Header Section */}
    <div className="border-b border-white/10 pb-8">
      <h2 className="text-4xl font-bold text-white flex items-center gap-4">
        <span className="w-2 h-10 bg-purple-500 rounded-full" />
        한국형 알고리즘 확증편향 진단 (K-Bias Diagnosis)
      </h2>
      <p className="mt-3 text-xl text-zinc-400 max-w-4xl">
        국내 유튜브 환경에서 가장 갈등이 심화된 <strong className="text-white">'정치'</strong> 이슈를 중심으로 알고리즘 편향을 심층 진단합니다.
      </p>
    </div>

    {/* Single Full-Size Module */}
    <div className="flex-1 w-full h-full">

      {/* MODULE A: Political Bias (Active) - Full Width */}
      <BentoCard
        title="정치적 성향 추이 (Political Bias Drift)"
        className="relative overflow-hidden border-purple-500/30 h-full"
        icon={Activity}
      >
        <div className="flex flex-col h-full gap-8">
          {/* Visualization: Bias Drift Chart (Time Series) */}
          <div className="flex-1 w-full relative rounded-2xl overflow-hidden border border-white/5 min-h-[500px] bg-black/40">
            {/* Gradient Defs */}
            <svg style={{ height: 0 }}>
              <defs>
                <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="50%" stopColor="#ef4444" stopOpacity={0} />
                  <stop offset="50%" stopColor="#3b82f6" stopOpacity={0} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.4} />
                </linearGradient>
              </defs>
            </svg>

            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={BIAS_DRIFT_DATA} margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.2} vertical={false} />
                <XAxis
                  dataKey="sequence"
                  name="Watch Order"
                  stroke="#71717a"
                  fontSize={14}
                  tickFormatter={(v) => `#${v}`}
                  dy={10}
                />
                <YAxis
                  domain={[-1.2, 1.2]}
                  stroke="#71717a"
                  fontSize={14}
                  ticks={[-1, 0, 1]}
                  tickFormatter={(v) => v === -1 ? 'Lib' : v === 1 ? 'Cons' : 'Neutral'}
                  dx={-10}
                />
                <RechartsTooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-black/90 text-lg border border-white/10 p-4 rounded-xl shadow-2xl">
                          <div className="text-zinc-400 mb-2">#{data.sequence} {data.title}</div>
                          <div className="font-mono font-bold text-xl">
                            Bias: <span className={data.bias > 0 ? "text-red-400" : "text-blue-400"}>{data.bias}</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />

                <ReferenceLine y={0} stroke="#71717a" strokeDasharray="3 3" strokeWidth={2} />

                {/* Area showing the drift magnitude */}
                <Area type="monotone" dataKey="bias" stroke="none" fill="url(#splitColor)" />

                {/* Line Path (No Dots, Thicker) */}
                <Line
                  type="monotone"
                  dataKey="bias"
                  stroke="#ffffff"
                  strokeWidth={5}
                  dot={false}
                  activeDot={{ r: 8, fill: '#fff' }}
                />
              </ComposedChart>
            </ResponsiveContainer>

            {/* Overlay Labels */}
            <div className="absolute top-6 right-6 text-xl font-bold text-red-500 bg-black/50 px-4 py-1 rounded">CONSERVATIVE (+1)</div>
            <div className="absolute bottom-6 right-6 text-xl font-bold text-blue-500 bg-black/50 px-4 py-1 rounded">LIBERAL (-1)</div>
          </div>

          {/* Analysis Report & Guide */}
          <div className="grid grid-cols-2 gap-8">
            <div className="p-6 rounded-2xl bg-purple-500/5 border border-purple-500/20">
              <h4 className="flex items-center gap-3 text-xl font-bold text-white mb-3">
                <BrainCircuit size={24} className="text-purple-400" />
                알고리즘 좌표 진단 (K-Bias Score)
              </h4>
              <p className="text-xl text-zinc-300 leading-relaxed">
                회원님의 확증 편향 지수는 <span className="text-red-400 font-bold text-2xl">0.96 (Danger)</span> 입니다.<br />
                <span className="text-base text-zinc-500 mt-2 block">
                  * 초기 중립 시청에서 시작되었으나, <strong>#60번째 영상</strong> 이후 급격한 우상향 패턴(Drift)이 고착화되었습니다.
                </span>
              </p>
            </div>

            {/* Interpretation Guide Box */}
            <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
              <h5 className="text-lg font-bold text-zinc-400 mb-4 flex items-center gap-2">
                <Info size={18} /> Bias Drift (성향 변화 추적)
              </h5>
              <ul className="space-y-3 text-sm text-zinc-400 leading-relaxed list-disc list-inside">
                <li>
                  <strong className="text-zinc-300">X축 (시간):</strong> 시청한 영상의 순서 100개입니다.
                </li>
                <li>
                  <strong className="text-zinc-300">Y축 (편향):</strong> 위쪽(+1)은 <span className="text-red-400">보수성향</span>, 아래쪽(-1)은 <span className="text-blue-400">진보성향</span> 강도를 의미합니다.
                </li>
                <li>
                  <strong className="text-white">우상향 패턴:</strong> 알고리즘 추천에 의해 점진적으로 한쪽 성향에 갇히는 <strong>'필터버블' 현상</strong>을 시각화했습니다.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </BentoCard>

    </div>
  </div>
);

/**
 * ============================================================================
 * [MAIN DASHBOARD] 
 * ============================================================================
 */
const Dashboard = () => {
  const [currentTab, setCurrentTab] = useState('profile'); // 'profile' | 'bias'

  return (
    <div className="flex min-h-screen font-sans text-zinc-100 bg-[#09090b] relative">
      {/* Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:32px_32px]"></div>

      {/* Sidebar */}
      <nav className="fixed left-0 top-0 h-full w-20 lg:w-64 border-r border-white/5 bg-black/40 backdrop-blur-xl flex flex-col p-4 z-50">
        <div className="mb-12 flex items-center gap-3 px-2 mt-4">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-[0_0_15px_#4f46e5]">
            <BrainCircuit size={18} className="text-white" />
          </div>
          <span className="hidden lg:block font-bold tracking-tight text-lg">DE-BIAS</span>
        </div>

        <div className="space-y-2">
          <button
            onClick={() => setCurrentTab('profile')}
            className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all ${currentTab === 'profile' ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            <LayoutDashboard size={20} />
            <span className="hidden lg:block text-sm font-medium">User Profile</span>
          </button>
          <button
            onClick={() => setCurrentTab('bias')}
            className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all ${currentTab === 'bias' ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            <ShieldAlert size={20} />
            <span className="hidden lg:block text-sm font-medium">Bias Diagnosis</span>
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 ml-20 lg:ml-64 p-8 relative z-0 overflow-y-auto h-screen">
        <header className="mb-8 flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              {currentTab === 'profile' ? 'User Profiling' : 'Bias Diagnosis'}
            </h1>
            <p className="text-zinc-400 text-sm">
              {currentTab === 'profile'
                ? 'Comprehensive analysis of watch history and interests.'
                : 'Deep learning analysis of political position and echo chamber risks.'}
            </p>
          </div>
          {/* User Badge */}
          <div className="flex items-center gap-3 bg-white/5 px-4 py-2 rounded-full border border-white/5">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-mono text-zinc-400">ANALYSIS COMPLETE</span>
          </div>
        </header>

        <motion.div
          key={currentTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="h-[calc(100vh-180px)]"
        >
          {currentTab === 'profile' ? <ProfileView /> : <BiasView />}
        </motion.div>
      </main>
    </div>
  );
};

export default Dashboard;