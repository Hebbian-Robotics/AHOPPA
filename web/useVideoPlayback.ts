import { type RefObject, useCallback, useEffect } from "react";
import { clearPlaybackAcceleration, setPlaybackAcceleration } from "./playback";

export function useVideoPlayback(
	videoReference: RefObject<HTMLVideoElement | null>,
	{
		playbackSpeed,
		sourcePlaybackSpeed,
		videoSource,
	}: {
		playbackSpeed: number;
		sourcePlaybackSpeed: number;
		videoSource: string | undefined;
	},
) {
	const initializePlayback = useCallback(
		(video: HTMLVideoElement) => {
			setPlaybackAcceleration(video, playbackSpeed / sourcePlaybackSpeed);
		},
		[playbackSpeed, sourcePlaybackSpeed],
	);

	useEffect(() => {
		if (!videoSource) return;
		const video = videoReference.current;
		if (video) initializePlayback(video);
		return () => {
			if (video) clearPlaybackAcceleration(video);
		};
	}, [videoSource, videoReference, initializePlayback]);

	useEffect(() => {
		if (!videoSource) return;
		const video = videoReference.current;
		let mounted = true;
		void customElements.whenDefined("video-skin").then(() => {
			if (mounted && video?.closest("video-skin")) video.controls = false;
		});
		return () => {
			mounted = false;
		};
	}, [videoSource, videoReference]);

	return initializePlayback;
}
