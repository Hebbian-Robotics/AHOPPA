import { Heading, Link, Text } from "@radix-ui/themes";
import { useQuery } from "@tanstack/react-query";
import {
	createElement,
	type RefObject,
	useCallback,
	useEffect,
	useRef,
	useState,
} from "react";
import { z } from "zod";
import { assetUrl } from "./assetUrl";
import { PlaybackSpeedControls, usePlaybackSpeed } from "./PlaybackControls";
import { useVideoPlayback } from "./useVideoPlayback";
import { VideoIdentifier } from "./VideoIdentifier";

const checkSchema = z.object({
	status: z.enum(["running", "completed", "failed"]),
	message: z.string(),
	video: z.string().nullable(),
	poster: z.string().nullable(),
});
interface SimulatorCheckProps {
	showMessage?: boolean;
	posterSource?: string;
	videoRef?: RefObject<HTMLVideoElement | null>;
	onPlaybackTimeChange?: (seconds: number) => void;
	videoIdentifier?: string;
	statusPath?: string;
	title?: string;
	description?: string;
	videoDescription?: string;
}

export function SimulatorCheck({
	showMessage = true,
	posterSource,
	videoRef,
	onPlaybackTimeChange,
	videoIdentifier = "isaac-sonic-reference-v1",
	statusPath = "isaac-reference-v1/status.json",
	title = "Isaac controller check",
	description = "G1 with the released SONIC v1.1 checkpoint. This check follows a saved walking reference; an LLM does not control it. The fixed-scene agent task is the next integration step.",
	videoDescription = "Playback uses simulation time. This unpaced controller test does not measure real-time agent performance. Yellow spheres show reference body targets; red markers indicate foot contacts.",
}: SimulatorCheckProps) {
	const videoReference = useRef<HTMLVideoElement>(null);
	const bindVideoReference = useCallback(
		(element: HTMLVideoElement | null) => {
			videoReference.current = element;
			if (videoRef) videoRef.current = element;
		},
		[videoRef],
	);
	const [playbackError, setPlaybackError] = useState(false);
	const { playbackSpeed, synchronizeNativePlayback } = usePlaybackSpeed();
	const check = useQuery({
		queryKey: ["isaac-integration-check", statusPath],
		queryFn: async ({ signal }) => {
			const response = await fetch(assetUrl(statusPath), {
				cache: "no-store",
				signal,
			});
			if (!response.ok) throw new Error("Controller check status unavailable");
			return checkSchema.parse(await response.json());
		},
		refetchInterval: (query) =>
			query.state.data?.status === "running" ? 10000 : false,
	});
	const video = check.data?.video;
	const initializePlayback = useVideoPlayback(videoReference, {
		playbackSpeed,
		sourcePlaybackSpeed: 1,
		videoSource: video ?? undefined,
	});

	useEffect(() => {
		const element = videoReference.current;
		if (
			!video ||
			!element ||
			!onPlaybackTimeChange ||
			!element.requestVideoFrameCallback
		)
			return;
		let callbackIdentifier = 0;
		const reportFrame = (
			_now: number,
			metadata: VideoFrameCallbackMetadata,
		) => {
			onPlaybackTimeChange(metadata.mediaTime);
			callbackIdentifier = element.requestVideoFrameCallback(reportFrame);
		};
		callbackIdentifier = element.requestVideoFrameCallback(reportFrame);
		return () => element.cancelVideoFrameCallback(callbackIdentifier);
	}, [video, onPlaybackTimeChange]);

	return (
		<section className="experiment-section" id={`video:${videoIdentifier}`}>
			<Heading as="h3" size="4">
				{title}
			</Heading>
			<Text as="p" size="2" color="gray" className="section-description">
				{description}
			</Text>
			{(showMessage || check.isError) && (
				<Text as="p" size="2" className="current-context" aria-live="polite">
					{check.isError
						? "Status unavailable. Reload to try again."
						: (check.data?.message ?? "Loading controller check…")}
				</Text>
			)}
			{video && (
				<article className="experiment-card">
					{createElement(
						"video-player",
						{ className: "experiment-player" },
						createElement(
							"video-skin",
							{},
							<video
								ref={bindVideoReference}
								src={assetUrl(video)}
								poster={
									posterSource ??
									(check.data?.poster ? assetUrl(check.data.poster) : undefined)
								}
								controls
								muted
								playsInline
								preload="none"
								data-source-playback-speed="1"
								aria-label={title}
								onLoadedMetadata={(event) =>
									initializePlayback(event.currentTarget)
								}
								onRateChange={(event) =>
									synchronizeNativePlayback(event.currentTarget)
								}
								onTimeUpdate={(event) => {
									if (!event.currentTarget.requestVideoFrameCallback)
										onPlaybackTimeChange?.(event.currentTarget.currentTime);
								}}
								onPause={(event) =>
									onPlaybackTimeChange?.(event.currentTarget.currentTime)
								}
								onSeeking={(event) =>
									onPlaybackTimeChange?.(event.currentTarget.currentTime)
								}
								onSeeked={(event) =>
									onPlaybackTimeChange?.(event.currentTarget.currentTime)
								}
								onError={() => setPlaybackError(true)}
							/>,
						),
					)}
					<PlaybackSpeedControls compact />
					<Text as="p" size="2" className="replay-caption">
						Offline replay of recorded motion. Use the speed controls to explore
						the full episode.
					</Text>
					<details className="recording-provenance">
						<summary>Recording details and direct video link</summary>
						<VideoIdentifier recordingIdentifier={videoIdentifier} />
						<Text as="p" size="1" color="gray">
							{videoDescription}
						</Text>
						<Link size="1" href={assetUrl(video)}>
							Open video
						</Link>
					</details>
					{playbackError && (
						<Text as="p" size="2" color="red">
							Video could not load. Try the direct video link.
						</Text>
					)}
				</article>
			)}
		</section>
	);
}
