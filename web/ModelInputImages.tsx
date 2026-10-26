import { Button, Heading, Link, Select, Table, Text } from "@radix-ui/themes";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { z } from "zod";
import { assetUrl } from "./assetUrl";

const inputImageSchema = z.object({
	label: z.string(),
	path: z.string(),
	sha256: z.string(),
});
const historyPairSchema = z.object({
	frames: z.array(
		z.object({
			observation_id: z.string(),
			action_id: z.string().nullable(),
			role: z.enum(["episode/start", "start", "midpoint", "current/end"]),
			captured_elapsed_seconds: z.number(),
			requested_elapsed_seconds: z.number(),
		}),
	),
	images: z.array(inputImageSchema).length(2),
});
const modelInputsSchema = z.object({
	provenance: z.string(),
	snapshots: z.array(
		z.object({
			identifier: z.string(),
			captured_wall_seconds: z.number(),
			request_wall_seconds: z.number(),
			images: z.array(inputImageSchema),
			pairs: z.array(historyPairSchema).optional(),
			previous_action_feedback: z
				.object({
					action_id: z.string(),
					tracking: z.record(
						z.string(),
						z.object({
							requested_position_m: z.array(z.number()).length(3),
							measured_position_m: z.array(z.number()).length(3),
							position_error_m: z.number(),
							orientation_error_degrees: z.number().optional(),
						}),
					),
				})
				.nullable()
				.optional(),
		}),
	),
});

function OriginalInputImage({
	image,
	requestIdentifier,
}: {
	image: z.infer<typeof inputImageSchema>;
	requestIdentifier: string;
}) {
	return (
		<figure className="observation-original-image">
			<figcaption>{image.label}</figcaption>
			<Link href={assetUrl(image.path)} target="_blank" rel="noreferrer">
				<img
					src={assetUrl(image.path)}
					alt={`${image.label} uploaded for ${requestIdentifier}`}
					loading="lazy"
				/>
			</Link>
		</figure>
	);
}

function PairedObservationHistory({
	pairs,
	requestIdentifier,
}: {
	pairs: z.infer<typeof historyPairSchema>[];
	requestIdentifier: string;
}) {
	const [pairIndex, setPairIndex] = useState(0);
	const [playing, setPlaying] = useState(false);
	useEffect(() => {
		if (!playing) return;
		const timeout = setTimeout(() => {
			if (pairIndex + 1 >= pairs.length) setPlaying(false);
			else setPairIndex(pairIndex + 1);
		}, 1000);
		return () => clearTimeout(timeout);
	}, [playing, pairIndex, pairs.length]);
	const pair = pairs[pairIndex];
	if (!pair) return null;
	return (
		<div className="observation-history-player">
			<Heading as="h4" size="3">
				Observation-history playback
			</Heading>
			<Text as="p" size="2">
				Discrete snapshots, one pair per second. This player is independent of
				the continuous motion replay.
			</Text>
			<div className="observation-history-controls">
				<Button
					variant="soft"
					disabled={pairIndex === 0}
					onClick={() => {
						setPlaying(false);
						setPairIndex(pairIndex - 1);
					}}
				>
					Previous pair
				</Button>
				<Button
					onClick={() => {
						if (!playing && pairIndex === pairs.length - 1) setPairIndex(0);
						setPlaying(!playing);
					}}
					disabled={pairs.length < 2}
				>
					{playing ? "Pause history" : "Play history"}
				</Button>
				<Button
					variant="soft"
					disabled={pairIndex === pairs.length - 1}
					onClick={() => {
						setPlaying(false);
						setPairIndex(pairIndex + 1);
					}}
				>
					Next pair
				</Button>
			</div>
			<Text as="p" size="2" aria-live="polite">
				Pair {pairIndex + 1} of {pairs.length}:{" "}
				{pair.frames
					.map(
						(frame) =>
							`${frame.role} · ${frame.captured_elapsed_seconds.toFixed(3)} s${frame.action_id ? ` · ${frame.action_id}` : ""}`,
					)
					.join("; ")}
			</Text>
			{pair.frames.length > 1 ? (
				<Text as="p" size="1" color="gray">
					These timestamps produced an identical image pair, shown once.
				</Text>
			) : null}
			<div className="observation-image-pair">
				{pair.images.map((image) => (
					<OriginalInputImage
						key={image.label}
						image={image}
						requestIdentifier={requestIdentifier}
					/>
				))}
			</div>
		</div>
	);
}

export function ModelInputImages({
	recordDirectory,
	decisionIdentifier,
}: {
	recordDirectory: string;
	decisionIdentifier: string | null;
}) {
	const [selectedIdentifier, setSelectedIdentifier] = useState("");
	const [expanded, setExpanded] = useState(false);
	const inputs = useQuery({
		queryKey: ["model-input-images", recordDirectory],
		enabled: expanded,
		queryFn: async ({ signal }) => {
			const response = await fetch(
				assetUrl(`${recordDirectory}/model-inputs.json`),
				{ signal },
			);
			if (!response.ok) throw new Error("Saved model inputs unavailable");
			return modelInputsSchema.parse(await response.json());
		},
		retry: false,
	});
	const snapshots = inputs.data?.snapshots ?? [];
	const snapshot = snapshots.find(
		(item) =>
			item.identifier ===
			(selectedIdentifier || decisionIdentifier || "initial"),
	);
	return (
		<details
			className="operator-model-inputs"
			onToggle={(event) => setExpanded(event.currentTarget.open)}
		>
			<summary>Images and feedback sent to the model</summary>
			<Text as="p" size="2">
				Original request images. Continuous motion replay contains additional
				frames; the model receives only the selected snapshots. Both camera
				views are supplied when listed here. Target overlays are presentation
				only.
			</Text>
			{snapshots.length > 0 ? (
				<>
					<Select.Root
						value={selectedIdentifier || "follow-playback"}
						onValueChange={(value) =>
							setSelectedIdentifier(value === "follow-playback" ? "" : value)
						}
					>
						<Select.Trigger aria-label="Choose model request images" />
						<Select.Content>
							<Select.Item value="follow-playback">
								Follow displayed motion
							</Select.Item>
							{snapshots.map((item) => (
								<Select.Item key={item.identifier} value={item.identifier}>
									{item.identifier} · {item.captured_wall_seconds.toFixed(1)} s
								</Select.Item>
							))}
						</Select.Content>
					</Select.Root>
					{snapshot ? (
						<>
							<Heading as="h3" size="3">
								Input for {snapshot.identifier}
							</Heading>
							<Text as="p" size="2">
								Current observation: {snapshot.captured_wall_seconds.toFixed(3)}{" "}
								s. Request group: {snapshot.request_wall_seconds.toFixed(2)} s.
								Images stay frozen while physics continues.
							</Text>
							{snapshot.pairs?.length ? (
								<PairedObservationHistory
									key={snapshot.identifier}
									pairs={snapshot.pairs}
									requestIdentifier={snapshot.identifier}
								/>
							) : (
								snapshot.images.map((image) => (
									<OriginalInputImage
										key={image.label}
										image={image}
										requestIdentifier={snapshot.identifier}
									/>
								))
							)}
							{snapshot.previous_action_feedback ? (
								<>
									<Heading as="h4" size="3">
										Previous action:{" "}
										{snapshot.previous_action_feedback.action_id}
									</Heading>
									<div className="observation-feedback-table">
										<Table.Root size="1">
											<Table.Header>
												<Table.Row>
													<Table.ColumnHeaderCell>
														Tracking point
													</Table.ColumnHeaderCell>
													<Table.ColumnHeaderCell>
														Requested xyz (m)
													</Table.ColumnHeaderCell>
													<Table.ColumnHeaderCell>
														Measured xyz (m)
													</Table.ColumnHeaderCell>
													<Table.ColumnHeaderCell>Error</Table.ColumnHeaderCell>
												</Table.Row>
											</Table.Header>
											<Table.Body>
												{Object.entries(
													snapshot.previous_action_feedback.tracking,
												).map(([name, feedback]) => (
													<Table.Row key={name}>
														<Table.RowHeaderCell>
															{name.replaceAll("_", " ")}
														</Table.RowHeaderCell>
														<Table.Cell>
															{feedback.requested_position_m
																.map((value) => value.toFixed(3))
																.join(", ")}
														</Table.Cell>
														<Table.Cell>
															{feedback.measured_position_m
																.map((value) => value.toFixed(3))
																.join(", ")}
														</Table.Cell>
														<Table.Cell>
															{feedback.position_error_m.toFixed(3)} m
															{feedback.orientation_error_degrees !== undefined
																? ` · ${feedback.orientation_error_degrees.toFixed(1)}°`
																: ""}
														</Table.Cell>
													</Table.Row>
												))}
											</Table.Body>
										</Table.Root>
									</div>
									<Text as="p" size="1" color="gray">
										Robot measurements in the command frame; errors alone do not
										establish contact, slip or a successful grasp.
									</Text>
								</>
							) : null}
						</>
					) : (
						<Text as="p" size="2">
							No exported input for this decision.
						</Text>
					)}
					<Text as="p" size="1" color="gray">
						{inputs.data?.provenance}
					</Text>
				</>
			) : (
				<Text as="p" size="2">
					{inputs.isPending
						? "Loading saved request images…"
						: "Exact request images have not been exported for this recording."}
				</Text>
			)}
		</details>
	);
}
