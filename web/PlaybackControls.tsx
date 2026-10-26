import { Button, Flex, Text } from "@radix-ui/themes";
import {
	createContext,
	type ReactNode,
	useCallback,
	useContext,
	useMemo,
	useRef,
	useState,
} from "react";
import { effectivePlaybackRate, setPlaybackAcceleration } from "./playback";

type PlaybackSettings = {
	playbackSpeed: number;
	changePlaybackSpeed: (speed: number) => void;
	synchronizeNativePlayback: (video: HTMLVideoElement) => void;
};
const PlaybackContext = createContext<PlaybackSettings | undefined>(undefined);

export function PlaybackSpeedProvider({ children }: { children: ReactNode }) {
	const [playbackSpeed, setPlaybackSpeed] = useState(1);
	const selectedSpeed = useRef(1);
	const changePlaybackSpeed = useCallback((speed: number) => {
		if (!Number.isFinite(speed) || speed <= 0) return;
		selectedSpeed.current = speed;
		setPlaybackSpeed(speed);
		for (const video of document.querySelectorAll("video")) {
			setPlaybackAcceleration(
				video,
				speed / Number(video.dataset.sourcePlaybackSpeed ?? "1"),
			);
		}
	}, []);
	const synchronizeNativePlayback = useCallback(
		(video: HTMLVideoElement) => {
			const speed =
				effectivePlaybackRate(video) *
				Number(video.dataset.sourcePlaybackSpeed ?? "1");
			if (Math.abs(speed - selectedSpeed.current) > 0.001)
				changePlaybackSpeed(speed);
		},
		[changePlaybackSpeed],
	);
	const settings = useMemo(
		() => ({ playbackSpeed, changePlaybackSpeed, synchronizeNativePlayback }),
		[playbackSpeed, changePlaybackSpeed, synchronizeNativePlayback],
	);
	return (
		<PlaybackContext.Provider value={settings}>
			{children}
		</PlaybackContext.Provider>
	);
}

export function usePlaybackSpeed() {
	const settings = useContext(PlaybackContext);
	if (!settings)
		throw new Error("Playback controls require a shared speed provider");
	return settings;
}

export function PlaybackSpeedControls({
	compact = false,
}: {
	compact?: boolean;
}) {
	const { playbackSpeed, changePlaybackSpeed } = usePlaybackSpeed();
	return (
		<Flex
			direction="column"
			gap="2"
			className={`playback-speed-controls ${compact ? "compact" : ""}`}
		>
			<Text size={compact ? "1" : "2"} weight="medium">
				Playback: {playbackSpeed}×
			</Text>
			<Flex gap="2" wrap="wrap" role="group" aria-label="Playback speed">
				{[1, 2, 4, 8, 16, 32].map((speed) => (
					<Button
						key={speed}
						size={compact ? "1" : "2"}
						color="gray"
						highContrast
						variant={playbackSpeed === speed ? "solid" : "outline"}
						aria-pressed={playbackSpeed === speed}
						onClick={() => changePlaybackSpeed(speed)}
					>
						{speed}×
					</Button>
				))}
			</Flex>
			{!compact && (
				<Text size="1" color="gray">
					One speed for the whole page. High speeds use frame skipping where
					required.
				</Text>
			)}
		</Flex>
	);
}
