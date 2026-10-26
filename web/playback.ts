const playbackCleanups = new WeakMap<HTMLVideoElement, () => void>();

export function clearPlaybackAcceleration(video: HTMLVideoElement) {
	playbackCleanups.get(video)?.();
	playbackCleanups.delete(video);
	delete video.dataset.acceleratedPlaybackRate;
}

export function effectivePlaybackRate(video: HTMLVideoElement): number {
	return Number(video.dataset.acceleratedPlaybackRate ?? video.playbackRate);
}

export function setPlaybackAcceleration(
	video: HTMLVideoElement,
	requestedRate: number,
) {
	clearPlaybackAcceleration(video);
	let nativeRate = requestedRate;
	while (true) {
		try {
			video.playbackRate = nativeRate;
			break;
		} catch (error) {
			if (
				!(error instanceof DOMException) ||
				error.name !== "NotSupportedError" ||
				nativeRate <= 1
			)
				throw error;
			nativeRate /= 2;
		}
	}
	if (nativeRate === requestedRate) return;
	// Browsers that cap native playback use seeking to skip additional frames.
	video.dataset.acceleratedPlaybackRate = String(requestedRate);
	let previousTime = performance.now();
	const timer = window.setInterval(() => {
		const currentTime = performance.now();
		const elapsedSeconds = Math.min((currentTime - previousTime) / 1000, 0.5);
		previousTime = currentTime;
		if (video.paused || video.ended || video.seeking || video.readyState < 2)
			return;
		video.currentTime = Math.min(
			video.duration,
			video.currentTime + elapsedSeconds * (requestedRate - nativeRate),
		);
	}, 250);
	const onRateChange = () => {
		if (video.playbackRate !== nativeRate) {
			clearPlaybackAcceleration(video);
			video.dispatchEvent(new Event("ratechange"));
		}
	};
	video.addEventListener("ratechange", onRateChange);
	playbackCleanups.set(video, () => {
		window.clearInterval(timer);
		video.removeEventListener("ratechange", onRateChange);
	});
	video.dispatchEvent(new Event("ratechange"));
}
