import { Heading, Text } from "@radix-ui/themes";
import costReport from "../docs/figures/api-costs.json";

const dollarFormatter = new Intl.NumberFormat("en-US", {
	style: "currency",
	currency: "USD",
});

function formatEstimatedCost(cost: {
	minimum_usd: string;
	maximum_usd: string;
}) {
	const minimum = dollarFormatter.format(Number(cost.minimum_usd));
	const maximum = dollarFormatter.format(Number(cost.maximum_usd));
	return minimum === maximum ? minimum : `${minimum} to ${maximum}`;
}

export function ApiCosts() {
	const unresolvedCalls =
		costReport.unreported_calls - costReport.recovered_calls;
	return (
		<div className="api-costs result-table">
			<Heading as="h3" size="5">
				Cost and episode duration
			</Heading>
			<Text as="p">
				All attempts averages all seven scheduled task attempts, including
				failures and early terminations. Successful attempts averages only
				attempts that met the full task success criterion. Episode duration is
				active elapsed time through the recorded cutoff; the all-closed control
				is excluded.
			</Text>
			<div className="result-table-scroll">
				<table>
					<caption>Mean API cost and elapsed time per attempt</caption>
					<thead>
						<tr>
							<th scope="col" rowSpan={2}>
								Model / effort
							</th>
							<th scope="colgroup" colSpan={2}>
								Estimated API cost (USD)
							</th>
							<th scope="colgroup" colSpan={2}>
								Episode time (min)
							</th>
						</tr>
						<tr>
							<th scope="col">All</th>
							<th scope="col">Successful</th>
							<th scope="col">All</th>
							<th scope="col">Successful</th>
						</tr>
					</thead>
					<tbody>
						{costReport.groups.map((group) => (
							<tr key={group.group}>
								<th scope="row">{group.label}</th>
								<td>
									{formatEstimatedCost(
										group.task_attempts.mean_cost_per_attempt,
									)}
								</td>
								<td>
									{group.successful_task_attempts ? (
										formatEstimatedCost(
											group.successful_task_attempts.mean_cost_per_attempt,
										)
									) : (
										<>
											—<sup>*</sup>
										</>
									)}
								</td>
								<td>
									{group.attempt_timing?.mean_elapsed_seconds == null
										? "—"
										: (group.attempt_timing.mean_elapsed_seconds / 60).toFixed(
												1,
											)}
								</td>
								<td>
									{group.successful_task_attempts ? (
										group.attempt_timing?.mean_successful_elapsed_seconds ==
										null ? (
											"—"
										) : (
											(
												group.attempt_timing.mean_successful_elapsed_seconds /
												60
											).toFixed(1)
										)
									) : (
										<>
											—<sup>*</sup>
										</>
									)}
								</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>
			<Text as="p" className="result-caveat">
				Astra high cost less in these runs despite generating more output
				tokens: cached reads covered about 50% of its input, compared with 29%
				at low. This matched low effort's success at an 18% lower mean cost.
			</Text>
			<Text as="p" className="result-caveat">
				Successful Sol high attempts cost less on average ($1.59 versus $1.96).
				Including spending on failed attempts, Sol low spent less per success:
				$3.00 versus $4.06. Cost per success divides total task-attempt spending
				by the success count.
			</Text>
			<Text as="p" className="result-caveat">
				* No successful attempt observed.
			</Text>
			<details>
				<summary>Pricing sources and calculation</summary>
				<p>
					We count each retained native API response once, including tool and
					correction calls. Cache reads, writes and uncached input are priced
					separately. Output includes reasoning tokens, counted once. Across all
					64 starts, {costReport.recorded_calls.toLocaleString("en-US")}{" "}
					responses have saved usage. Provider reports and rejection policies
					resolve {costReport.recovered_calls} additional calls;{" "}
					{unresolvedCalls} Gemini calls remain unresolved, about 0.30% of all
					attempted calls. Their charges are excluded. Four HTTP 400 rejections
					are confirmed not charged. GPU hosting, taxes and account discounts
					are excluded from all estimates.
				</p>
				<p>
					We match OpenAI usage to recorded completion timestamps and Anthropic
					usage to timestamped HTTP receipts, then attribute only residual usage
					that uniquely matches an interrupted attempt. Google monitoring does
					not expose the cached-input split needed to price the remaining Gemini
					calls reliably.
				</p>
				<p>
					Rates:{" "}
					<a href="https://developers.openai.com/api/docs/pricing">OpenAI</a>,{" "}
					<a href="https://platform.claude.com/docs/en/models/opus-5-5/overview">
						Anthropic
					</a>
					,{" "}
					<a href="https://ai.google.dev/gemini-api/docs/pricing#gemini-4-argon">
						Google
					</a>
					.{" "}
					<a href="https://github.com/Hebbian-Robotics/AHOPPA/blob/main/docs/api-costs.md">
						Reproduce the estimates
					</a>
					.
				</p>
			</details>
		</div>
	);
}
