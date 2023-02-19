const defaultTheme = require("tailwindcss/defaultTheme");

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["InterVariable", "Inter", ...defaultTheme.fontFamily.sans],
        mono: ["JetBrains Mono", ...defaultTheme.fontFamily.mono],
      },
      colors: {
        bleu: "#3273d9",
      },
      boxShadow: {
        "nb-sm": "2px 1px 0 #000",
        "nb-base": "4px 4px 0 #000",
        "nb-xl": "14px 14px 0 #000",
      },
      typography: {
        DEFAULT: {
          css: {
            "code::before": {
              content: '""',
            },
            "code::after": {
              content: '""',
            },
            a: {
              textDecorationLine: "none",
            },
            "a:hover": {
              textDecorationLine: "underline",
            },
            "p, li": {
              code: {
                backgroundColor: "#d4d4d4",
                padding: "0.1rem 0.2rem",
                borderRadius: "0.250rem",
                fontWeight: "300",
                color: "#171717",
              },
            },
            blockquote: {
              borderLeftColor: "#3273d9",
            },
          },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
