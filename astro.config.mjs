import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";
import cloudflare from "@astrojs/cloudflare";

// https://astro.build/config
export default defineConfig({
  site: "https://kpthe.dev",
  markdown: {
    syntaxHighlight: "prism",
  },
  integrations: [
    mdx(),
    sitemap({
      customPages: [
        "https://kpthe.dev/",
        "https://kpthe.dev/blog/",
        "https://kpthe.dev/projects/",
      ],
    }),
    tailwind(),
  ],
  output: "server",
  adapter: cloudflare(),
});
