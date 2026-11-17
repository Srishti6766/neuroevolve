/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        obsidian: '#080B14',
        slatecard: '#111827',
        violetglow: '#8B5CF6',
        cyanglow: '#06B6D4',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        'violet-glow': '0 0 24px rgba(139, 92, 246, 0.22)',
        'cyan-glow': '0 0 24px rgba(6, 182, 212, 0.22)',
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(to right, rgba(139, 92, 246, 0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(139, 92, 246, 0.06) 1px, transparent 1px)',
      },
    },
  },
  plugins: [],
}
