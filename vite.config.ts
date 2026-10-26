import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import sirv from "sirv";
import { defineConfig } from "vite";

const siteBase = process.env.HUMANOID_AGENT_SITE_BASE ?? "/frontier-motion/";

export default defineConfig({
	root: "web",
	base: siteBase,
	publicDir: "public",
	plugins: [
		react(),
		{
			name: "local-experiment-records",
			configureServer(server) {
				for (const directory of ["transfer"]) {
					server.middlewares.use(
						`${siteBase}${directory}`,
						sirv(resolve("site", directory), { dev: true }),
					);
				}
			},
		},
	],
	build: {
		outDir: "dist",
		emptyOutDir: true,
		rolldownOptions: {
			input: {
				main: resolve("web/index.html"),
			},
		},
	},
});
