import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Deploy prefix: "/" standalone (the HF Space), "/play/" when the bundle
  // is served under the Microduck Academy (VITE_BASE=/play/ npm run build).
  // Runtime asset URLs are base-relative ("./assets", "./policies") so
  // only the bundle references in index.html depend on this.
  base: process.env.VITE_BASE || "/",
  plugins: [react()],
  // game.js imports the npm engines and their WASM sidecars with Vite ?url.
  // They are emitted under bundle/ and served from the app's own origin.
  server: {
    port: 5173,
  },
  build: {
    // Keep the JS/CSS bundle out of dist/assets/: the game's static assets
    // (public/assets/) land there and must keep their historical URLs.
    assetsDir: "bundle",
    chunkSizeWarningLimit: 1500,
  },
});
