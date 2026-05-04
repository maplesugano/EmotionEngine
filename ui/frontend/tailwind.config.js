/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        booth: {
          bg: "#0b0b12",
          panel: "#15151f",
          panel2: "#1d1d2a",
          edge: "#2a2a3c",
          ink: "#e7e7ef",
          muted: "#8a8aa0",
          accent: "#7c5cff",
          accent2: "#22d3ee",
          warm: "#f59e0b",
          cool: "#3b82f6",
          good: "#34d399",
          bad: "#f87171",
        },
      },
      boxShadow: {
        booth: "0 6px 30px -10px rgba(124,92,255,0.25), 0 1px 0 rgba(255,255,255,0.04) inset",
      },
    },
  },
  plugins: [],
};
