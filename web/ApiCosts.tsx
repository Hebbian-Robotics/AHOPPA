import { Heading, Text } from "@radix-ui/themes";
import costReport from "../docs/figures/api-costs.json";

const dollarFormatter = new Intl.NumberFormat("en-US", {
	style: "currency",
	currency: "USD",
});

export function ApiCosts() {
	return (
		<div className="api-costs result-table">
			<Heading as="h3" size="5">
				Estimated API cost
			</Heading>
			<Text as="p">
				Mean cost per scheduled task attempt, using recorded token usage and
				published standard API rates. Each row covers the same seven attempts as
				the success comparison, including unsuccessful attempts. The all-closed
				control is excluded.
			</Text>
			<div className="result-table-scroll">
				<table>
					<caption>Estimated API cost per task attempt in US dollars</caption>
					<thead>
						<tr>
							<th scope="col">Model / effort</th>
							<th scope="col">USD / attempt</th>
							<th scope="col">Calls without response usage</th>
						</tr>
					</thead>
					<tbody>
						{costReport.groups.map((group) => {
							const costs = group.task_attempts.mean_cost_per_attempt;
							const minimum = dollarFormatter.format(Number(costs.minimum_usd));
							const maximum = dollarFormatter.format(Number(costs.maximum_usd));
							return (
								<tr key={group.group}>
									<th scope="row">{group.label}</th>
									<td>
										{minimum === maximum ? minimum : `${minimum} to ${maximum}`}
										{group.task_attempts.unreported_calls > 0 ? (
											<sup>*</sup>
										) : null}
									</td>
									<td>{group.task_attempts.unreported_calls}</td>
								</tr>
							);
						})}
					</tbody>
				</table>
			</div>
			<Text as="p" className="result-caveat">
				* Calls without recorded response usage are excluded from the estimate,
				so actual charges may be higher. OpenAI's Usage API confirms that cache
				writes used the 30-minute rate. These are token-priced estimates, not
				billed amounts; GPU hosting, taxes and account discounts are excluded.
			</Text>
			<details>
				<summary>Pricing sources and calculation</summary>
				<p>
					We count each retained native API response once, including tool and
					correction calls. Cache reads, writes and uncached input are priced
					separately. Output includes reasoning tokens, counted once. Across all
					64 starts, {costReport.recorded_calls.toLocaleString("en-US")}{" "}
					responses have usage and {costReport.unreported_calls} calls have no
					recorded response usage.
				</p>
				<p>
					Astra's Usage API totals match all 996 retained responses exactly.
					Sol's report contains six additional calls, matching the six
					interrupted calls without saved responses. Their aggregate usage is
					documented, but it is excluded from per-attempt estimates because we
					cannot assign it to individual attempts reliably.
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
