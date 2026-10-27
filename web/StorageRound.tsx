import { Heading, Text } from "@radix-ui/themes";
import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { z } from "zod";
import storageFacilityMap from "../docs/figures/storage-facility-map.png";
import closeDrawerImage from "../docs/figures/storage-observations/close-d.png";
import returnHomeImage from "../docs/figures/storage-observations/home.png";
import inspectDrawerImage from "../docs/figures/storage-observations/inspect-d.png";
import storageResults from "../docs/figures/storage-results.json";
import { ApiCosts } from "./ApiCosts";
import { assetUrl } from "./assetUrl";
import { ModelInputImages } from "./ModelInputImages";
import { parseShutdownResults, ShutdownResults } from "./ShutdownResults";
import { SimulatorCheck } from "./SimulatorCheck";

const recordingSchema = z.object({
	identifier: z.uuid(),
	title: z.string(),
	statusPath: z.string().startsWith("transfer/"),
	description: z.string(),
	videoDescription: z.string(),
	durationSeconds: z.number().positive(),
	phases: z.array(z.object({ name: z.string(), at: z.number().nonnegative() })),
	modelInputsDirectory: z.string().startsWith("transfer/").optional(),
	decisions: z
		.array(
			z.object({
				identifier: z.string(),
				at: z.number().nonnegative(),
				progress: z.string(),
				subtasks: z.array(
					z.object({
						identifier: z.string(),
						description: z.string(),
						status: z.string(),
					}),
				),
			}),
		)
		.optional(),
});
const manifestSchema = z.object({ records: z.array(recordingSchema) });

function EpisodeRecording({
	record,
}: {
	record: z.infer<typeof recordingSchema>;
}) {
	const videoReference = useRef<HTMLVideoElement>(null);
	const [playbackSeconds, setPlaybackSeconds] = useState(0);
	const activePhase = record.phases.reduce(
		(active, phase, index) => (phase.at <= playbackSeconds ? index : active),
		-1,
	);
	const activeDecision = record.decisions
		?.slice()
		.reverse()
		.find((decision) => decision.at <= playbackSeconds);
	return (
		<article className="episode-recording">
			<SimulatorCheck
				showMessage={false}
				posterSource={
					record.statusPath.includes("/storage-cmp3-astra-low-open-a-d-r1/")
						? returnHomeImage
						: undefined
				}
				videoRef={videoReference}
				onPlaybackTimeChange={setPlaybackSeconds}
				videoIdentifier={record.identifier}
				statusPath={record.statusPath}
				title={record.title}
				description={record.description}
				videoDescription={record.videoDescription}
			/>
			<details className="episode-inspector">
				<summary>
					Explore the plan, decision timeline and original model inputs
				</summary>
				<div className="transfer-phases">
					<Heading as="h3" size="3">
						Episode timeline
					</Heading>
					<Text as="p" size="2">
						{record.durationSeconds.toFixed(1)} s recorded. Select a phase to
						seek.
					</Text>
					<ol className="transfer-phase-list">
						{record.phases.map((phase, index) => (
							<li key={`${phase.name}-${phase.at}`}>
								<button
									type="button"
									aria-current={index === activePhase ? "step" : undefined}
									onClick={() => {
										if (videoReference.current)
											videoReference.current.currentTime = phase.at;
										setPlaybackSeconds(phase.at);
									}}
								>
									<span>{phase.name.replaceAll("_", " ")}</span>
									<span>{phase.at.toFixed(1)} s</span>
								</button>
							</li>
						))}
					</ol>
				</div>
				{record.decisions && (
					<div className="transfer-plan">
						<Heading as="h3" size="3">
							Model's current plan
						</Heading>
						<Text as="p" size="2">
							Updated when a completed response arrives. Subtask status and
							progress come from the model. The evaluator scores task outcomes
							independently.
						</Text>
						{activeDecision ? (
							<>
								<Text as="p">{activeDecision.progress}</Text>
								<ol>
									{activeDecision.subtasks.map((subtask) => (
										<li key={subtask.identifier}>
											{subtask.description} ({subtask.status})
										</li>
									))}
								</ol>
							</>
						) : (
							<Text as="p">No completed model response at this time.</Text>
						)}
					</div>
				)}
				{record.modelInputsDirectory && (
					<ModelInputImages
						recordDirectory={record.modelInputsDirectory}
						decisionIdentifier={activeDecision?.identifier ?? null}
					/>
				)}
			</details>
		</article>
	);
}

export function EpisodeShowcase() {
	const manifest = useQuery({
		queryKey: ["storage-round-recordings"],
		queryFn: async ({ signal }) => {
			const response = await fetch(assetUrl("transfer/manifest.json"), {
				cache: "no-store",
				signal,
			});
			if (!response.ok) throw new Error("Episode recordings unavailable");
			return manifestSchema.parse(await response.json());
		},
		refetchInterval: 30000,
		retry: false,
	});
	const featuredRecording =
		manifest.data?.records.find((record) =>
			record.statusPath.includes("/storage-cmp3-astra-low-open-a-d-r1/"),
		) ?? manifest.data?.records[0];
	return (
		<section
			id="recordings"
			className="episode-showcase"
			aria-label="Cabinet inspection recordings"
		>
			<div className="showcase-introduction">
				<Heading as="h2" size="5">
					A complete cabinet inspection
				</Heading>
				<Text as="p">
					One instruction starts the inspection. The model chooses its route,
					checks the drawers and returns home. The full episode is shown without
					cuts, with adjustable playback speed.
				</Text>
			</div>
			{featuredRecording ? (
				<>
					<div id={`full-episode-${featuredRecording.identifier}`}>
						<EpisodeRecording
							key={featuredRecording.identifier}
							record={featuredRecording}
						/>
					</div>
					{manifest.data?.records
						.filter(
							(record) => record.identifier !== featuredRecording.identifier,
						)
						.map((record) => (
							<details
								className="additional-recording"
								id={`full-episode-${record.identifier}`}
								key={record.identifier}
							>
								<summary>{record.title}</summary>
								<EpisodeRecording record={record} />
							</details>
						))}
				</>
			) : (
				<Text as="p" className="recording-placeholder" role="status">
					{manifest.isError
						? "Recordings are unavailable. The results and original observation figure are below."
						: "Loading the recorded cabinet inspection…"}
				</Text>
			)}
		</section>
	);
}

export function StorageRound() {
	return (
		<>
			<section id="task" className="task-section">
				<div className="task-description">
					<Heading as="h2" size="7">
						Inspect the room. Leave every drawer closed.
					</Heading>
					<Text as="p">
						The G1 inspects the top drawer of four cabinets on a facility map,
						pushes any open drawer closed by hand, leaves closed drawers alone
						and returns home. The independent evaluator scores drawer closure,
						hand contact and the return hold separately from the model's
						completion declaration.
					</Text>
					<Text as="p">
						A block of tall SimReady storage lockers stands between home and
						cabinets A and B, and cabinets C and D stand behind home, so no
						drawer front is visible from the start. The model must use the map
						to plan where to look.
					</Text>
					<details className="scene-details">
						<summary>Scene and controller details</summary>
						<Text as="p">
							The robot stands on the standard flat ground collider under the
							room&apos;s visual floor: the room&apos;s thin floor collider made
							the released SONIC planner spin and fall under commands it tracked
							on flat ground and in upstream&apos;s MuJoCo sim2sim. Raw facing
							changes are approached gradually, with the commanded facing held
							within 15° of the measured heading, as the released gamepad turns
							by small stick increments.
						</Text>
					</details>
				</div>
				<figure className="facility-map">
					<Heading as="h4" size="3">
						Map supplied to the model
					</Heading>
					<a href={storageFacilityMap}>
						<img
							src={storageFacilityMap}
							width={770}
							height={770}
							loading="lazy"
							alt="Top-down storage bay map with home at the origin facing right toward a 2.1 m tall block of storage lockers. Cabinets A and B stand on the east wall behind the lockers; cabinets C and D stand on the west wall behind home. Red lines mark the drawer fronts."
						/>
					</a>
					<figcaption>
						The exact image included in every cabinet inspection request. The
						map shows fixed cabinet locations. The model uses its camera views
						to locate the robot and check drawers. Select the image to open it
						at full size.
					</figcaption>
				</figure>
				<div className="observation-sequence">
					<figure>
						<a href={inspectDrawerImage}>
							<img
								src={inspectDrawerImage}
								width="960"
								height="540"
								loading="lazy"
								alt="Original onboard view of cabinet D with its top drawer open."
							/>
						</a>
						<figcaption>
							<strong>Find an open drawer</strong>
							<span>Onboard observation at 98 s</span>
						</figcaption>
					</figure>
					<figure>
						<a href={closeDrawerImage}>
							<img
								src={closeDrawerImage}
								width="960"
								height="540"
								loading="lazy"
								alt="Original onboard view as the G1 pushes cabinet D's drawer shut by hand."
							/>
						</a>
						<figcaption>
							<strong>Push it shut by hand</strong>
							<span>Onboard observation at 193 s</span>
						</figcaption>
					</figure>
					<figure>
						<a href={returnHomeImage}>
							<img
								src={returnHomeImage}
								width="960"
								height="540"
								loading="lazy"
								alt="Original external view of the G1 upright at home near the end of the episode."
							/>
						</a>
						<figcaption>
							<strong>Return home</strong>
							<span>External observation at 942 s</span>
						</figcaption>
					</figure>
					<p>
						Original images supplied to GPT-6 Astra during the featured episode.
						Select an image to view it at full resolution.
					</p>
				</div>
			</section>
			<section id="results" className="results-section">
				<div className="section-heading">
					<Heading as="h2" size="7">
						Results across four frontier models
					</Heading>
					<Text as="p">
						An initial comparison of GPT-6 Astra, GPT-6.1 Sol, Claude Opus 5.5
						and Gemini 4 Argon.
					</Text>
				</div>
				<div className="result-finding">
					<Heading as="h3" size="5">
						GPT-6 Astra completed 12 of 14 task attempts.
					</Heading>
					<Text as="p">
						Successful task attempts took 615 to 1492 seconds. The model chose
						inspection routes, pushed drawers shut and returned home through
						online commands to a fixed controller, without a task skill library.
						Eleven of its twelve successes viewed all four drawer fronts.
					</Text>
				</div>
				<dl className="study-design">
					<div>
						<dt>64 scheduled episodes</dt>
						<dd>Every start retained</dd>
					</div>
					<div>
						<dt>8 drawer conditions</dt>
						<dd>Including one condition with all drawers closed</dd>
					</div>
					<div>
						<dt>2 reasoning settings</dt>
						<dd>Low and high effort</dd>
					</div>
					<div>
						<dt>Simulation only</dt>
						<dd>One attempt per model, effort and condition</dd>
					</div>
				</dl>
				<ShutdownResults
					results={parseShutdownResults(storageResults)}
					identifier="storage-results-title"
					title="Model comparison"
					introduction="Four frontier models each attempted eight drawer conditions at low and high reasoning effort with an 1800 s budget. The conditions were none open, each single cabinet, and A+D, B+C and A+B. We froze the seeded schedule before launch. All 43 closures were by hand, and every initially closed drawer stayed closed."
					outcomeCaption="Each bar accounts for seven scheduled task attempts with at least one drawer initially open. Success uses all seven starts, including API-error starts. It requires every drawer closed and settled, the model's completion declaration and a supported standing hold at home. API errors, falls, time limits and incorrect declarations have separate outcome categories. The all-closed control appears in the table. All recorded closures were by hand, and no initially closed drawer was disturbed. The startup fall occurred before the model's first command."
					caveat="Lag rarely coincided with falls. The results include episodes that exceeded the timing limit. Each model, effort and condition has one attempt on one layout; repeated trials are needed to estimate reliability."
				/>
				<div className="result-finding">
					<Heading as="h3" size="5">
						Higher effort can improve progress without improving success
					</Heading>
					<Text as="p">
						Sol high closed 9 of 10 initially open drawers, compared with 4 at
						low effort, and reached physical completion in 5 of 7 attempts
						versus 4. Two physical completions went unreported before a timeout
						or fall, leaving fewer successful attempts: 3 versus 4. Higher
						effort increased decision latency roughly two to four times across
						models in this comparison.
					</Text>
				</div>
				<div className="result-finding">
					<Heading as="h3" size="5">
						Falls also occurred without recorded drawer contact
					</Heading>
					<Text as="p">
						Among eight all-closed controls, three succeeded, four fell and one
						had an API error. Across all conditions, 16 of 19 falls had no
						recorded drawer contact; other collisions remain possible. Eleven of
						18 post-command falls ended in holds commanding zero translational
						speed with retained pose and facing targets. Prolonged target holds
						during inference may contribute to instability; this is a
						hypothesis, not a demonstrated cause.
					</Text>
				</div>
				<ApiCosts />
			</section>
		</>
	);
}
