import { Button, Flex, Link, Text } from "@radix-ui/themes";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { copyVideoIdentifier } from "./clipboard";
import { videoUuid } from "./videoIdentifiers";

export function VideoIdentifier({
	recordingIdentifier,
}: {
	recordingIdentifier: string;
}) {
	const [copyStatus, setCopyStatus] = useState("idle");
	const identifier = useQuery({
		queryKey: ["video-uuid", recordingIdentifier],
		queryFn: () => videoUuid(recordingIdentifier),
		staleTime: Number.POSITIVE_INFINITY,
	});
	const anchor = identifier.data ? `video:${identifier.data}` : undefined;
	useEffect(() => {
		if (
			anchor &&
			[`#${anchor}`, `#${encodeURIComponent(anchor)}`].includes(location.hash)
		) {
			const element = document.getElementById(anchor);
			for (
				let parent = element?.parentElement;
				parent;
				parent = parent.parentElement
			) {
				if (parent instanceof HTMLDetailsElement) parent.open = true;
			}
			element?.scrollIntoView({ block: "start" });
		}
	}, [anchor]);
	return (
		<Flex
			id={anchor}
			data-video-id={identifier.data}
			gap="2"
			align="center"
			wrap="wrap"
			className="video-identifier"
		>
			<Text size="1" color="gray">
				Video ID
			</Text>
			<code>
				{identifier.data ?? (identifier.isError ? "Unavailable" : "Loading…")}
			</code>
			<Button
				size="1"
				variant="soft"
				color="gray"
				disabled={!identifier.data}
				onClick={async () => {
					if (identifier.data) {
						setCopyStatus(
							(await copyVideoIdentifier(identifier.data)) ? "copied" : "error",
						);
					}
				}}
			>
				{copyStatus === "copied" ? "Copied" : "Copy ID"}
			</Button>
			{anchor && (
				<Link size="1" href={`#${encodeURIComponent(anchor)}`}>
					Direct link
				</Link>
			)}
			{copyStatus === "error" && (
				<Text size="1" role="status">
					Select the ID above to copy it.
				</Text>
			)}
		</Flex>
	);
}
