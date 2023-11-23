/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/javascript/**/*.{js,ts,jsx,tsx,mdx}',
    './src/javascript/component/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'colour': '#4D1D3B', // RGB(77, 29, 59)
      },
    },
  },
  plugins: [],
}

