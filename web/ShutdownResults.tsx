import { Heading, Text } from "@radix-ui/themes";
import { type ReactNode, useState } from "react";
import { z } from "zod";

const outcomeSchema = z.enum([
	"success",
	"declared_without_success",
	"fell",
	"startup_fall",
	"time_limit",
	"infrastructure_error",
]);
type EpisodeOutcome = z.infer<typeof outcomeSchema>;

const resultsSchema = z.object({
	schema_version: z.literal("webolt-shutdown-results-v1"),
	episode_budget_seconds: z.number().positive(),
	outcomes: z.array(outcomeSchema),
	providers: z.array(
		z.object({
			provider: z.string(),
			label: z.string(),
			manipulation: z.object({
				started: z.number().int(),
				completed: z.number().int(),
				outcomes: z.record(outcomeSchema, z.number().int().nonnegative()),
				physical_completion_observed: z.number().int(),
				open_drawers: z.number().int(),
				open_drawers_closed: z.number().int(),
				declared_without_physical_completion: z.number().int(),
				successful_completion_seconds: z.array(z.number().nonnegative()),
				all_drawers_viewed: z.number().int().optional(),
			}),
			control: z.object({
				started: z.number().int(),
				completed: z.number().int(),
				task_success: z.number().int(),
			}),
			median_request_seconds: z.number().positive(),
			attempt_timing: z
				.object({
					mean_elapsed_seconds: z.number().nonnegative().nullable(),
					mean_successful_elapsed_seconds: z.number().nonnegative().nullable(),
					median_decision_seconds: z.number().nonnegative().nullable(),
					median_successful_decision_seconds: z
						.number()
						.nonnegative()
						.nullable(),
				})
				.nullable()
				.optional(),
		}),
	),
	timing: z.object({
		completed_episodes: z.number().int(),
		timing_valid_episodes: z.number().int(),
		median_peak_physics_lag_seconds: z.number(),
		maximum_peak_physics_lag_seconds: z.number(),
	}),
});

export type ShutdownResultsData = z.infer<typeof resultsSchema>;

export function parseShutdownResults(source: unknown): ShutdownResultsData {
	return resultsSchema.parse(source);
}

function outcomeLabel(outcome: EpisodeOutcome, budgetSeconds: number): string {
	return {
		success: "Success",
		declared_without_success: "Declared, criterion not met",
		fell: "Fell",
		startup_fall: "Startup fall (before first command)",
		time_limit: `${budgetSeconds} s limit`,
		infrastructure_error: "API error",
	}[outcome];
}

type HoveredMark = { text: string; x: number; y: number } | null;

function useMarkTooltip() {
	const [hoveredMark, setHoveredMark] = useState<HoveredMark>(null);
	const markProperties = (text: string) => ({
		tabIndex: 0,
		"aria-label": text,
		onPointerEnter: (event: React.PointerEvent<HTMLElement>) => {
			const bounds = event.currentTarget.getBoundingClientRect();
			setHoveredMark({
				text,
				x: bounds.left + bounds.width / 2,
				y: bounds.top,
			});
		},
		onFocus: (event: React.FocusEvent<HTMLElement>) => {
			const bounds = event.currentTarget.getBoundingClientRect();
			setHoveredMark({
				text,
				x: bounds.left + bounds.width / 2,
				y: bounds.top,
			});
		},
		onPointerLeave: () => setHoveredMark(null),
		onBlur: () => setHoveredMark(null),
	});
	const tooltip = hoveredMark && (
		<div
			className="chart-tooltip"
			role="status"
			style={{ left: hoveredMark.x, top: hoveredMark.y }}
		>
			{hoveredMark.text}
		</div>
	);
	return { markProperties, tooltip };
}

function OutcomeChart({
	results,
	caption,
}: {
	results: ShutdownResultsData;
	caption: ReactNode;
}) {
	const { markProperties, tooltip } = useMarkTooltip();
	const budget = results.episode_budget_seconds;
	return (
		<figure className="result-chart">
			<Heading as="h3" size="4">
				Task outcomes
			</Heading>
			<ul className="chart-legend" aria-label="Episode outcomes">
				{results.outcomes.map((outcome) => (
					<li key={outcome}>
						<span className={`chart-swatch outcome-${outcome}`} />
						{outcomeLabel(outcome, budget)}
					</li>
				))}
			</ul>
			<div className="outcome-rows">
				{results.providers.map((provider) => (
					<div className="outcome-row" key={provider.provider}>
						<span className="chart-row-label">{provider.label}</span>
						<div className="outcome-bar">
							{results.outcomes
								.filter(
									(outcome) => provider.manipulation.outcomes[outcome] > 0,
								)
								.map((outcome) => {
									const count = provider.manipulation.outcomes[outcome];
									return (
										<span
											key={outcome}
											className={`outcome-segment outcome-${outcome}`}
											style={{ flexGrow: count }}
											{...markProperties(
												`${provider.label}: ${outcomeLabel(outcome, budget)}, ${count} of ${provider.manipulation.started} starts`,
											)}
										/>
									);
								})}
						</div>
						<span className="chart-row-value">
							{provider.manipulation.outcomes.success}/
							{provider.manipulation.started} successful attempts
						</span>
					</div>
				))}
			</div>
			{tooltip}
			<figcaption>{caption}</figcaption>
		</figure>
	);
}

function CompletionTimeChart({ results }: { results: ShutdownResultsData }) {
	const { markProperties, tooltip } = useMarkTooltip();
	const budget = results.episode_budget_seconds;
	const tickStep = budget > 900 ? 600 : 300;
	const ticks = Array.from(
		{ length: Math.floor(budget / tickStep) + 1 },
		(_, index) => index * tickStep,
	);
	return (
		<figure className="result-chart">
			<Heading as="h3" size="4">
				Successful task completion and decision latency
			</Heading>
			<div className="outcome-row timing-column-labels" aria-hidden="true">
				<span />
				<span>Successful completion time</span>
				<span>
					Decision latency (s)
					<br />
					All / successful attempts
				</span>
			</div>
			<div className="timeline-rows">
				{results.providers.map((provider) => (
					<div className="outcome-row" key={provider.provider}>
						<span className="chart-row-label">{provider.label}</span>
						<div className="timeline-track">
							{provider.manipulation.successful_completion_seconds.map(
								(seconds) => (
									<span
										key={seconds}
										className="timeline-dot"
										style={{ left: `${(seconds / budget) * 100}%` }}
										{...markProperties(
											`${provider.label}: success declared at ${seconds.toFixed(0)} s`,
										)}
									/>
								),
							)}
							{provider.manipulation.successful_completion_seconds.length ===
								0 && <span className="timeline-empty">No successes</span>}
						</div>
						<span className="chart-row-value">
							{provider.attempt_timing ? (
								<>
									{provider.attempt_timing.median_decision_seconds?.toFixed(
										1,
									) ?? "—"}{" "}
									/{" "}
									{provider.attempt_timing.median_successful_decision_seconds?.toFixed(
										1,
									) ?? "—"}{" "}
									s
								</>
							) : (
								`${provider.median_request_seconds.toFixed(1)} s / decision`
							)}
						</span>
					</div>
				))}
				<div className="outcome-row" aria-hidden="true">
					<span className="chart-row-label" />
					<div className="timeline-ticks">
						{ticks.map((tick) => (
							<span
								key={tick}
								className="timeline-tick"
								style={{ left: `${(tick / budget) * 100}%` }}
							>
								{tick} s
							</span>
						))}
					</div>
					<span className="chart-row-value" />
				</div>
			</div>
			{tooltip}
			<figcaption>
				Elapsed wall time at the successful completion declaration, out of the{" "}
				{budget} s budget. The right column reports median model-decision
				latency over all task attempts and over successful attempts only. Each
				value is the median of per-attempt medians for completed decisions.
				Physics keeps running during inference.
			</figcaption>
		</figure>
	);
}

function ResultTable({ results }: { results: ShutdownResultsData }) {
	return (
		<details className="result-table">
			<summary>Table view</summary>
			<div className="result-table-scroll">
				<table>
					<thead>
						<tr>
							<th scope="col">Model</th>
							<th scope="col">Success / scheduled</th>
							<th scope="col">Task outcomes available</th>
							<th scope="col">Physical completion observed</th>
							<th scope="col">Open drawers closed</th>
							<th scope="col">All 4 drawers seen</th>
							<th scope="col">Declared without completion</th>
							<th scope="col">Falls</th>
							<th scope="col">{results.episode_budget_seconds} s limit</th>
							<th scope="col">API errors</th>
							<th scope="col">All drawers initially closed</th>
						</tr>
					</thead>
					<tbody>
						{results.providers.map(
							({ provider, label, manipulation, control }) => (
								<tr key={provider}>
									<th scope="row">{label}</th>
									<td>
										{manipulation.outcomes.success}/{manipulation.started}
									</td>
									<td>
										{manipulation.completed} of {manipulation.started}
									</td>
									<td>{manipulation.physical_completion_observed}</td>
									<td>{manipulation.open_drawers_closed}</td>
									<td>{manipulation.all_drawers_viewed ?? "–"}</td>
									<td>{manipulation.declared_without_physical_completion}</td>
									<td>{manipulation.outcomes.fell}</td>
									<td>{manipulation.outcomes.time_limit}</td>
									<td>{manipulation.outcomes.infrastructure_error}</td>
									<td>
										{control.task_success}/{control.started}
									</td>
								</tr>
							),
						)}
					</tbody>
				</table>
			</div>
			<Text as="p" size="2">
				Success uses all scheduled task attempts. Physical completion,
				inspection and incorrect declarations are observed counts from completed
				episodes; API errors have no inferred physical outcome.
			</Text>
		</details>
	);
}

export function ShutdownResults({
	results,
	identifier,
	title,
	introduction,
	strip,
	outcomeCaption,
	caveat,
}: {
	results: ShutdownResultsData;
	identifier: string;
	title: string;
	introduction: ReactNode;
	strip?: ReactNode;
	outcomeCaption: ReactNode;
	caveat: ReactNode;
}) {
	const { timing } = results;
	return (
		<section className="shutdown-results" aria-labelledby={identifier}>
			<Heading as="h3" size="5" id={identifier}>
				{title}
			</Heading>
			<Text as="p">{introduction}</Text>
			{strip}
			<OutcomeChart results={results} caption={outcomeCaption} />
			<CompletionTimeChart results={results} />
			<ResultTable results={results} />
			<Text as="p" size="2" className="result-caveat">
				Peak physics lag had a median of{" "}
				{(timing.median_peak_physics_lag_seconds * 1000).toFixed(0)} ms (maximum{" "}
				{(timing.maximum_peak_physics_lag_seconds * 1000).toFixed(0)} ms); only{" "}
				{timing.timing_valid_episodes} of {timing.completed_episodes} completed
				episodes met the 100 ms limit. {caveat}
			</Text>
		</section>
	);
}
