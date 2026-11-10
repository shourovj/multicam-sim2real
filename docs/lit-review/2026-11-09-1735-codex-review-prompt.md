# Codex Review Prompt

The prompt given to both Codex runs ([Sol](2026-11-09-1741-lit-survey-codex-sol.md), [Astra](2026-11-09-1744-lit-survey-codex-astra.md)), verbatim. Each run's working directory held only this prompt and a copy of the [v0 design doc](../2026-11-09-1416-design-doc-v0.md). To keep the second opinion independent, neither run saw the Claude survey or review.

---

You are being asked for an independent second opinion on a research project, plus a literature survey. Today is 2026-11-09. Work read-only: do not modify any files; put everything in your final answer.

## Context from the project lead (paraphrased from what they told another agent)

"I am looking to do some literature survey. I think the basic idea is that we want to see if text-to-video models, video generation models, or world models can be improved by using an intermediate simulator step. We want to check whether there's existing work in the field and whether there are good pairings of simulators with video generation or world models. We want to compare them against baselines that generate video from text alone and from text plus a first frame or something like that. This is just going to be an initial draft or a scoping experiment."

- The v0 design doc is in `design-doc-v0.md` in your working directory. Read it. It is the team's (AFRL Thread 1) earlier plan; some of it is superseded by the notes below.
- Notes the lead added at the end of the design doc (Google Doc version):
  - Main things we need: simulator choice; T2V models that can work with the simulator; metrics that can judge their performance; literature review of this field ("I know of CoDeR": https://becauseimbatman0.github.io/CoDeR — "Code Plans, Diffusion Renders: Open-Ended Generative World Modeling").
  - What we already have: one-line experiments showing that current SOTA world-model / T2V models cannot faithfully simulate motion on their own, especially when that motion goes out of frame.
  - By Tuesday 3 PM: the best pairing of simulator <> WM/T2V/X2V; the big players of the field (e.g., CoDeR). Start the paper/report doc by Wednesday 3 PM.
- Scope clarifications from the lead:
  - For now, just 3 test videos: (1) drones take off from the ground and move somewhere; (2) drones just moving in the air; (3) satellite view of a convoy going along a road and then taking a right onto a field.
  - Drones are the main subject and clearly visible on screen (no tiny <3 px targets for now).
  - Survey the general field first, then narrow to drones/satellite.
  - Open question from the lead: "Can video generation models directly take a simulated scene with its motion as input, and not necessarily a rendered video? I am thinking that feeding a rendered video is more like style transfer and not necessarily video generation / world modeling."
  - The lead also wants the experiments to reveal how well these models work in practice, as opposed to how they are advertised.
- Compute: 8x A100 80GB (no RT cores) on Linux servers. No model training planned for v0. Deadline for v0 results: Friday 2026-11-13.

## What to do

Use subagents to parallelize the literature review (one focused topic per subagent). Search the web; verify every paper link resolves to the right paper; never invent papers; mark anything you could not verify as UNVERIFIED. Prefer 2024-2026 work.

Return one Markdown report with:

1. **Literature survey**: the most relevant prior work (aim for 25-40 entries total), grouped however you think best. For each: title, first author, year, venue, verified link, what the intermediate/simulator step is, what is passed to the video model, baselines and key result, and its relevance to us.
2. **The lead's open question** (scene state vs rendered video): your answer with evidence, and which open models can take simulator state directly.
3. **Simulator <> generator pairings**: concrete recommendation for each of the 3 test videos, with setup cost and risks given our hardware.
4. **Your honest critique of the project**: is the core idea sound and worth doing? Is the text-only / text+first-frame baseline comparison fair or informative? What is novel vs already known? What would make it a strong result vs a trivial one? Be direct; disagree where warranted.
5. **Your proposed experiment plan for this week** (by Friday): conditions, models, metrics, deliverable, and what to cut if time runs short.
6. A short list of what you are least sure about.
