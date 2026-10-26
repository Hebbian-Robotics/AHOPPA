import { Heading, Link, Text } from "@radix-ui/themes";
import overviewFigure from "../docs/figures/system-overview.svg";
import paperPdf from "../paper/paper.pdf?url";
import { PlaybackSpeedProvider } from "./PlaybackControls";
import { EpisodeShowcase, StorageRound } from "./StorageRound";
import cmuWordmark from "./template/assets/cmu-wordmark.svg";
import paperIcon from "./template/assets/file-pdf.svg";
import githubIcon from "./template/assets/github.svg";
import hebbianWordmark from "./template/assets/hebbian-wordmark.svg";

function ProjectPage() {
	return (
		<main className="page-shell">
			<a className="skip-link" href="#results">
				Skip to results
			</a>
			<header className="page-header">
				<Heading as="h1" className="page-title publication-title">
					AHOPPA
				</Heading>
				<Text as="p" className="publication-subtitle">
					Agentic Humanoid Operation through Perception, Planning, and Action
				</Text>
				<Text as="p" className="publication-authors">
					<Link href="https://www.linkedin.com/in/kingston-kuan/">
						Kingston Kuan<sup>1</sup>
					</Link>
					{" · "}
					<Link href="https://www.linkedin.com/in/samuel-j-chew/">
						Samuel J Chew<sup>2</sup>
					</Link>
					{" · "}
					<Link href="https://www.linkedin.com/in/ongjjbrandon/">
						Brandon Ong<sup>1</sup>
					</Link>
				</Text>
				<Text as="p" className="publication-affiliations">
					<sup>1</sup>Hebbian Robotics · <sup>2</sup>Carnegie Mellon University
				</Text>
				<div className="affiliation-logos">
					<img
						className="affiliation-wordmark"
						src={hebbianWordmark}
						alt="Hebbian Robotics"
						width="144"
						height="44"
					/>
					<a
						href="https://www.cmu.edu/"
						aria-label="Carnegie Mellon University"
					>
						<img
							className="cmu-wordmark"
							src={cmuWordmark}
							alt="Carnegie Mellon University"
							width="216"
							height="19"
						/>
					</a>
				</div>
				<nav className="publication-links" aria-label="Project resources">
					<a className="publication-link" href={paperPdf}>
						<img src={paperIcon} width="16" height="16" alt="" />
						<span>Paper</span>
					</a>
					<a
						className="publication-link"
						href="https://github.com/Hebbian-Robotics/AHOPPA"
					>
						<img src={githubIcon} width="16" height="16" alt="" />
						<span>Code</span>
					</a>
					<a
						className="publication-link"
						href="https://huggingface.co/datasets/hebbianrobotics/AHOPPA"
					>
						Raw data
					</a>
					<a
						className="publication-link publication-link-secondary"
						href="#recordings"
					>
						Videos
					</a>
				</nav>
				<Text as="p" className="intro">
					A humanoid working across a room must decide where to go, interpret
					what it sees and adjust its movements as the task unfolds. AHOPPA
					studies whether frontier multimodal models can make these decisions
					through visual feedback and a fixed motion controller in simulation.
					We report results obtained in the last few weeks, following recent
					model releases, on a cabinet inspection task.
				</Text>
			</header>
			<figure className="overview-figure project-overview">
				<a href={overviewFigure}>
					<img
						src={overviewFigure}
						alt="One task instruction starts a loop of model observations, a revisable plan, numerical head and wrist targets, SONIC motion control and fresh visual feedback. Open the full-resolution figure."
						width="1800"
						height="740"
						fetchPriority="high"
						decoding="async"
					/>
				</a>
				<figcaption>
					The model observes, updates its plan and sends head and hand targets
					with locomotion commands to SONIC. New observations close the loop.
					Scene images come from one successful GPT-6 Astra episode.
					<a className="figure-link" href={overviewFigure}>
						View full-size overview
					</a>
				</figcaption>
			</figure>
			<EpisodeShowcase />
			<nav className="section-navigation" aria-label="On this page">
				<a href="#task">The task</a>
				<a href="#results">Results</a>
				<a href="#method">How it works</a>
				<a href="#scope">Setup and limits</a>
			</nav>
			<div className="research-content">
				<StorageRound />
				<section id="method" className="method-section">
					<div className="section-heading">
						<Heading as="h2" size="7">
							From visual feedback to head and hand targets
						</Heading>
						<Text as="p">
							The model chooses what to do next. SONIC tracks its head and hand
							targets and handles locomotion while the simulated world keeps
							running.
						</Text>
					</div>
					<div className="method-grid">
						<div>
							<Heading as="h3" size="4">
								Observe and plan
							</Heading>
							<Text as="p">
								The model receives paired onboard RGB-D and fixed external RGB,
								with a short action history and measured robot feedback from
								encoders, IMU and calibrated kinematics. Read-only depth queries
								measure selected pixels of the current onboard image. The study
								uses torso and external cameras. Object poses and evaluator
								state remain private to the simulator.
							</Text>
						</div>
						<div>
							<Heading as="h3" size="4">
								Generate motion targets
							</Heading>
							<Text as="p">
								The model sends head and wrist poses with native SONIC walking
								and posture commands. It chooses direction, speed, facing and
								tracking targets; SONIC supplies the gait and balance control.
								The interface approaches facing changes gradually, following the
								released gamepad's small turns. The model plans each reach and
								inspection from the observations it receives.
							</Text>
						</div>
						<div>
							<Heading as="h3" size="4">
								Act with continuous feedback
							</Heading>
							<Text as="p">
								Newton GPU physics advances during model inference. Fresh
								observations inform each decision, and an independent evaluator
								checks the physical task after the model declares completion.
							</Text>
						</div>
					</div>
				</section>
				<section id="scope" className="study-section">
					<Heading as="h2" size="6">
						Setup and limits
					</Heading>
					<Text as="p">
						The cabinet inspection task uses NVIDIA&apos;s released Simple Room,
						Isaac Lab&apos;s Sektion cabinets, SimReady storage lockers, Newton
						GPU physics and the native SONIC controller. A static facility map
						guides inspection. The model infers drawer states from its camera
						views. We record closures, disturbances, falls, the return to home,
						completion declarations and elapsed time.
					</Text>
					<Text as="p">
						All experiments are in simulation. The comparison runs one episode
						per model, effort and condition on one layout. Repetitions,
						older-model baselines and hardware trials are needed to measure
						reliability, capability gains and transfer.
					</Text>
				</section>
			</div>
			<footer className="publication-footer">
				<p>
					Website adapted from the{" "}
					<Link href="https://github.com/nerfies/nerfies.github.io">
						Nerfies template
					</Link>{" "}
					and licensed under{" "}
					<Link href="https://creativecommons.org/licenses/by-sa/4.0/">
						CC BY-SA 4.0
					</Link>
					. See the{" "}
					<Link href={`${import.meta.env.BASE_URL}assets/website-LICENSE.txt`}>
						website licenses
					</Link>
					.
				</p>
			</footer>
		</main>
	);
}

export function App() {
	return (
		<PlaybackSpeedProvider>
			<ProjectPage />
		</PlaybackSpeedProvider>
	);
}
