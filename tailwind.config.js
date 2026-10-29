/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        gis: {
          bg: '#080c14',
          surface: '#0d1322',
          card: '#121a2d',
          border: '#1e293f',
          hover: '#18233c',
          muted: '#64748b',
          accent: '#06b6d4',      // Cyan / Sensor glow
          emerald: '#10b981',     // Vegetation / Safe
          sprawl: '#f97316',      // Sprawl expansion / Urban alert
          danger: '#ef4444',      // Deforestation / Loss
          purple: '#8b5cf6',      // Deep learning / Classification
          amber: '#f59e0b',       // High sprawl risk
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'gis-glow': '0 0 20px -5px rgba(6, 182, 212, 0.25)',
        'sprawl-glow': '0 0 20px -5px rgba(249, 115, 22, 0.25)',
      }
    },
  },
  plugins: [],
}
