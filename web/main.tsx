import "@radix-ui/themes/styles.css";
import "@videojs/html/video/skin.css";
import "./template/nerfies.css";
import "./styles.css";
import { Theme } from "@radix-ui/themes";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoot } from "react-dom/client";
import { App } from "./App";

const queryClient = new QueryClient({
	defaultOptions: { queries: { refetchOnWindowFocus: false } },
});
const root = document.getElementById("root");
if (root)
	createRoot(root).render(
		<QueryClientProvider client={queryClient}>
			<Theme accentColor="gray" grayColor="slate" radius="small" scaling="100%">
				<App />
			</Theme>
		</QueryClientProvider>,
	);

// Native video controls remain usable if the enhanced player cannot load.
void import("./media").catch(() =>
	console.warn(
		"Enhanced media controls unavailable; using native video controls.",
	),
);
