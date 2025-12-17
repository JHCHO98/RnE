// HyperspeedPresets.js
export const hyperSpeedOptions = {
    onSpeedUp: () => { },
    onSlowDown: () => { },
    distortion: 'turbulentDistortion', // 알고리즘의 혼란함을 표현
    length: 400,
    roadWidth: 9,
    islandWidth: 2,
    lanesPerRoad: 3,
    fov: 90,
    fovSpeedUp: 150,
    speedUp: 2.2, // 가속 시 더 빠르게
    carLightsFade: 0.4,
    totalSideLightSticks: 50,
    lightPairsPerRoadWay: 50,
    shoulderLinesWidthPercentage: 0.05,
    brokenLinesWidthPercentage: 0.1,
    brokenLinesLengthPercentage: 0.5,
    lightStickWidth: [0.12, 0.5],
    lightStickHeight: [1.3, 1.7],
    movingAwaySpeed: [60, 80],
    movingCloserSpeed: [-120, -160],
    carLightsLength: [400 * 0.05, 400 * 0.15],
    carLightsRadius: [0.05, 0.14],
    carWidthPercentage: [0.3, 0.5],
    carShiftX: [-0.8, 0.8],
    carFloorSeparation: [0, 5],
    colors: {
        roadColor: 0x080808, // Zinc-950 (Deep Black)
        islandColor: 0x0a0a0a,
        background: 0x09090b, // Background Black
        shoulderLines: 0x27272a, // Zinc-800
        brokenLines: 0x52525b, // Zinc-600
        // Project Theme Colors
        leftCars: [0x8b5cf6, 0x6366f1, 0xa855f7], // Purples (Users' Intent)
        rightCars: [0xef4444, 0x10b981, 0xffffff], // Red(Danger), Green(Safe), White(Neutral)
        sticks: 0x6366f1, // Indigo Sticks
    }
};