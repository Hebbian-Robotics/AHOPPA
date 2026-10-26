import { Heading, Text } from "@radix-ui/themes";
import { useRef, useState } from "react";
import successPoster from "./highlights/astra-inspection.jpg";
import successVideo from "./highlights/astra-inspection.mp4?url";
import successCaptions from "./highlights/astra-inspection.vtt?url";
import failurePoster from "./highlights/opus-inspection.jpg";
import failureVideo from "./highlights/opus-inspection.mp4?url";
import failureCaptions from "./highlights/opus-inspection.vtt?url";
import highlightProvenance from "./highlights/provenance.json";
import { PlaybackSpeedControls, usePlaybackSpeed } from "./PlaybackControls";
import { useVideoPlayback } from "./useVideoPlayback";

const highlightAssets = {
	astra: {
		video: successVideo,
		poster: successPoster,
		captions: successCaptions,
		title: "GPT-6 Astra cabinet inspection highlights",
		recordingIdentifier: "4eeec4e1-3f8b-54f6-9bcd-5220642c8893",
	},
	opus: {
		video: failureVideo,
		poster: failurePoster,
		captions: failureCaptions,
		title: "Claude Opus 5.5 cabinet inspection highlights",
		recordingIdentifier: "684ea327-f0fd-5598-8d42-427d4486ada8",
	},
};

function FullEpisodeLink({
	provider,
}: {
	provider: keyof typeof highlightAssets;
}) {
	const recordingIdentifier = highlightAssets[provider].recordingIdentifier;
	return (
		<a
			href={`#video:${recordingIdentifier}`}
			onClick={() => {
				const disclosure = document.getElementById(
					`full-episode-${recordingIdentifier}`,
				);
				if (disclosure instanceof HTMLDetailsElement) disclosure.open = true;
			}}
		>
			Full episode and original model inputs
		</a>
	);
}

export function HighlightVideo({
	provider,
}: {
	provider: keyof typeof highlightAssets;
}) {
	const assets = highlightAssets[provider];
	const videoReference = useRef<HTMLVideoElement>(null);
	const [playbackSeconds, setPlaybackSeconds] = useState(0);
	const { playbackSpeed, synchronizeNativePlayback } = usePlaybackSpeed();
	const initializePlayback = useVideoPlayback(videoReference, {
		playbackSpeed,
		sourcePlaybackSpeed: highlightProvenance.source_speed,
		videoSource: assets.video,
	});
	const provenance = highlightProvenance.records.find(
		(record) => record.provider === provider,
	);
	if (!provenance)
		throw new Error(`Missing highlight provenance for ${provider}`);
	const activeSegment = provenance.segments.reduce(
		(active, segment) =>
			segment.output_start_seconds <= playbackSeconds ? segment : active,
		provenance.segments[0],
	);
	const episodeSeconds = Math.min(
		activeSegment.source_end_seconds,
		activeSegment.source_start_seconds +
			Math.max(0, playbackSeconds - activeSegment.output_start_seconds) *
				highlightProvenance.source_speed,
	);
	return (
		<figure className="highlight-video">
			<video
				ref={videoReference}
				src={assets.video}
				poster={assets.poster}
				controls
				muted
				playsInline
				preload="none"
				data-source-playback-speed={highlightProvenance.source_speed}
				aria-label={assets.title}
				onLoadedMetadata={(event) => {
					initializePlayback(event.currentTarget);
					setPlaybackSeconds(event.currentTarget.currentTime);
				}}
				onRateChange={(event) => synchronizeNativePlayback(event.currentTarget)}
				onTimeUpdate={(event) =>
					setPlaybackSeconds(event.currentTarget.currentTime)
				}
				onSeeked={(event) =>
					setPlaybackSeconds(event.currentTarget.currentTime)
				}
			>
				<track
					kind="captions"
					src={assets.captions}
					srcLang="en"
					label="Episode events"
				/>
			</video>
			<p className="highlight-event">
				<strong>{activeSegment.label}</strong>
				<span className="highlight-clock">
					Episode {episodeSeconds.toFixed(0)} s
				</span>
			</p>
			<PlaybackSpeedControls compact />
			<figcaption>
				Selected excerpts with cuts, shown at the selected playback speed. Left:
				external view; right: onboard view. Offline pose replay; these frames
				were not model observations.
			</figcaption>
		</figure>
	);
}

export function EpisodeComparison() {
	return (
		<section
			className="episode-comparison"
			aria-labelledby="episode-comparison-title"
		>
			<Heading as="h3" size="5" id="episode-comparison-title">
				A completion claim still needs a physical check
			</Heading>
			<Text as="p" className="comparison-introduction">
				Two selected attempts with A and D initially open, both at low effort.
				Clips advance independently. The aggregate results below retain every
				scheduled start.
			</Text>
			<div className="episode-comparison-grid">
				<article>
					<Heading as="h4" size="4">
						GPT-6 Astra: task completed
					</Heading>
					<HighlightVideo provider="astra" />
					<p>
						A and D closed; the upright home hold passed. The model declared
						completion at 955 s.
					</p>
					<p className="comparison-timing">
						Peak physics lag: 31 ms, within the 100 ms limit.
					</p>
					<FullEpisodeLink provider="astra" />
				</article>
				<article>
					<Heading as="h4" size="4">
						Claude Opus 5.5: completion rejected
					</Heading>
					<HighlightVideo provider="opus" />
					<p>
						A closed, but D remained 2.8 cm open, beyond the 2 cm threshold. The
						home hold also failed. The model declared completion at 1034 s.
					</p>
					<p className="comparison-timing">
						Peak physics lag: 189 ms, exceeding the 100 ms limit.
					</p>
					<FullEpisodeLink provider="opus" />
				</article>
			</div>
		</section>
	);
}
