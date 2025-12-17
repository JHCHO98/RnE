/** @type {import('tailwindcss').Config} */
module.exports = {
  // 여기가 중요합니다! src 폴더 내의 모든 js, jsx, ts, tsx 파일을 바라보게 설정
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      // 폰트 설정 (없어도 작동은 함)
      fontFamily: {
        sans: ['Pretendard', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      // 색상 팔레트가 혹시 없을 경우를 대비해 확장
      colors: {
        // 기본 Tailwind 색상을 사용하므로 별도 추가 없어도 됨
      }
    },
  },
  plugins: [],
}