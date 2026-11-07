# Video demo of the T-ITS paper (BH-CBF)

This folder holds the scripts of the supplementary video of the paper *Safeguarding Multi-Agent
Reinforcement Learning for Connected and Automated Vehicles with Recursively Feasible Control Barrier
Functions*. The video also helps explain the filters' behavior in
the evaluation to readers who don't know the code. Everything the scripts
produce or cache lives in **`outputs/tits26/video_demo/`** (OUT below):

- **Final video:** `OUT/bh_cbf_video_demo.mp4` (1920 × 1080, 30 fps, H.264).
- **Raw scene videos for slides:** `OUT/raw/` (map panels only, without titles, captions, or fades).

The video follows the paper's **four evaluation stages**: (1) pre-training with safety rewards,
(2) zero-shot deployment with run-time filters, (3) filter-in-the-loop fine-tuning, and (4) the
comparison with human drivers. The evaluation's output folders keep the names of the earlier
three-stage numbering: `stage1_zero_shot` and `stage1_filters_ttc_iter100` hold Stages 1 and 2,
`stage2_finetune` Stage 3, and `stage3_human_*` Stage 4. The same holds for the cache folder
`OUT/stage2/` (Stage 3 rollouts) and the tails `s3_*` (Stage 4).

The video **replays rollouts of the evaluation; it simulates nothing while rendering**. Some
inputs that the evaluation did not store (braking tails, fine-tuning rollouts) were produced once by
rerunning deployments; they are cached in `OUT/tails/` and `OUT/stage2/`. If OUT is lost, the
commands in "Recipes" (and the docstring of `video.py`) rebuild it.

## Scripts (this folder, `sigmarl/evaluation_tits26/video_demo/`)

| Script | Role |
|---|---|
| `video.py` | Renders every scene (`render`) and concatenates the clips (`assemble`). All scene choices are constants at its top. |
| `tails.py` | Reruns a BH-CBF deployment exactly as the evaluation ran it, and saves the braking tails (backup rollout until standstill) at every step → `OUT/tails/`. |
| `rollouts.py` | Saves Stage 3 (fine-tuning) rollouts (checkpoints deployed without a filter) → `OUT/stage2/`. It works on a copy of the run, so the paper's run folders stay untouched. |
| `repair.py` | Reruns a BH-CBF deployment and records the repair rounds of the certified filter (QP constraints of the first QP, candidates, violated (e, t), added QP constraints, fallback) of every filter problem → `OUT/repair/`. `list` prints the filter problems with repairs. |
| `select_windows.py` | Lists candidate time windows (collision criteria; see "How the scenes were chosen"). |
| `hero.py` | Renders the background loop of the project page's header (bassamlab.github.io/BH-CBF): the `backup_round` scene, full-bleed (1920 × 800), no text, the last second crossfaded into the first → `OUT/hero/hero_round.{mp4,jpg}`. |

Paths in the scripts are relative to the repository root, so run everything from
the repository root and set
`SIGMARL_AGENT_PROFILE=full_scale` (full-scale maps). The examples use this shell function (zsh or
bash):

```bash
R() { SIGMARL_AGENT_PROFILE=full_scale python -m "sigmarl.evaluation_tits26.video_demo.$1" "${@:2}"; }
R video render                                   # all clips (~2 min) -> OUT/clips/, OUT/raw/
R video render --only stage4 stage4_card         # only some clips (names: see the table below)
R video render --preview --only stage4           # one PNG per clip (its middle frame) -> OUT/previews/
R video assemble                                 # OUT/clips/*.mp4 -> OUT/bh_cbf_video_demo.mp4 (no re-encoding)
```

After changing a scene, re-render only its clip and run `assemble`. `assemble` concatenates every
`OUT/clips/*.mp4`, so delete the clip files of renamed or removed clips first. To inspect a frame, use
`ffmpeg -ss <seconds> -i OUT/clips/<clip>.mp4 -frames:v 1 frame.png`.

**Requirements:**
- `ffmpeg`.
- The font Carlito, which has the same letter widths as Calibri. Set `SIGMARL_CARLITO_DIR` to the
  folder of its `.ttf` files (default: the fonts bundled with LibreOffice on macOS). Without it,
  matplotlib falls back to another font.

## Output folder (`outputs/tits26/video_demo/`)

| Path | Content |
|---|---|
| `bh_cbf_video_demo.mp4` | The final video. |
| `clips/NN_<name>.mp4` | One clip per scene (with fades); `list.txt` is the concat list of `assemble`. |
| `raw/<scene>.mp4` | Map panels of each scene, cropped to the panels and their titles, no fades, for reuse in slides. Written during `render` (`Clip.start_raw`/`stop_raw`). |
| `previews/` | PNGs of `render --preview`. |
| `hero/` | Background loop of the project page's header and its poster (`hero.py`). |
| `assets/` | Cover-page images: `logo_i11_rwth.png`, `logo_unibw.png` (composited on white and cropped to the logo; also used by the closing card), and `photo_xu.jpg`, `photo_sayed.jpg`, `photo_alrifaee.jpg`. |
| `tails/*.npz` | Braking tails of Stage 2 runs and counterfactual masters: `steps` [S] and `tails` [S, H, n, 3] (pose per tail step and vehicle; padded with the standstill pose). Step indices are those of the source rollout. |
| `tails/s3_*.pkl` | Braking tails of Stage 4 episodes (used only by the extra side-by-side clip): `captured` = list of (recording time [s], {track id: tail [H, 3]}) for the active vehicles, plus `deviation` (rerun vs. paper episode; it was 0.0 m for all four). |
| `tails/_sim/` | Outputs of the reruns (logs, the rerun Stage 4 episodes). Not used by the renderer. |
| `stage2/<run>/` | Copy of the Stage 3 (fine-tuning) run config and checkpoints, and the rollouts `out_td_iter{0,100}_nofilter_seed{201..205}.td`. |
| `logs/` | Logs of the background jobs. |

## Scenes (`CLIPS` in `video.py`)

Paths in the source column: `assets/`, `tails/`, and `stage2/` are in OUT; Stage 2 runs are in
`outputs/tits26/stage1_filters_ttc_iter100/<policy>/<map>/`.

| # | Clip | Content | Source data | Key constants |
|---|---|---|---|---|
| 00 | `title` | Cover: title, authors, logos, photos, affiliations as in the paper (Xu: RWTH; Sayed and Alrifaee: University of the Bundeswehr Munich). | `assets/` | `scene_title` |
| 01 | `contributions` | "Contributions": the lead line "We propose the Backup-Horizon Control Barrier Function (BH-CBF) with the following key properties:" and five numbered points (aligned with the paper's abstract). | – | `scene_contributions` |
| 02 | `backup_round` | Key idea: the braking tails of all vehicles, roundabout, steps 250–350 at real time. | Stage 2 run `round…ttc_seed_1…/ours/dec/seed201` + `tails/backup_round.npz` | `BACKUP_ROUND`, window in `CLIPS` |
| 03 | `backup_ind` | Same, intersection, steps 40–140. | `ind…ttc_seed_4…/ours/dec/seed202` + `tails/backup_ind.npz` | `BACKUP_IND` |
| 04 | `stage1_card` | Section card (introduces TTC). | – | `CLIPS` |
| 05 | `stage1_rewards` | TTC / one-step CBF / BH-CBF-informed reward: policies of training seed 1 on the roundabout (iteration 500), deployed without a filter from the same start, first 25 s at 2× speed. Collisions: 2 / 3 / 0. | `outputs/tits26/stage1_zero_shot/round_location0__stage1_{ttc,onestep,bh}_seed_1/round_location0/unfiltered/cen/seed203` | `REWARDS*` |
| 06 | `stage2_card` | Section card. | – | `CLIPS` |
| 07 | `stage2_counterfactual` | Four scenes (paper fig. footprints (a), (b), (e), (f)). Left: the unsafe policy alone, paused at its collision. Right: the same joint state with BH-CBF and its trails, continuing for 5 s (scenes 1–2) or 3 s (scenes 3–4) after the collision. | `outputs/tits26/stage1_counterfactual/*/seed301.npz` + `tails/cf_{a..d}.npz` | `COUNTERFACTUAL_SCENES`, `CONTINUE` |
| 08 | `stage2_comparison` | No filter / one-step CBF (the paper's A-Condition) / BH-CBF (with trails), same policy and start, roundabout, first 25 s at 2× speed. Collisions: 3 / 2 / 0. | Stage 2 runs `round…ttc_seed_4…/{unfiltered/cen,ours_onestep/dec,ours/dec}/seed202` + `tails/cmp_round.npz` | `COMPARISON*` |
| 09 | `stage3_card` | Section card. | – | `CLIPS` |
| 10 | `stage3_finetuning` | Roundabout policy (TTC seed 1, iteration 100) before and after fine-tuning on the intersection with BH-CBF in the loop. Both panels without a filter, so no trails. First 30 s at 2× speed. Collisions: 3 / 0. | `stage2/…/out_td_iter{0,100}_nofilter_seed205.td` | `FINETUNE_*` |
| 11 | `stage4_card` | Section card (112 windows, 1264 vehicles, over 115 km). | – | `CLIPS` |
| 12 | `stage4` | Four scenes (paper fig. human-footprints; the paper keeps only (c) and (d), the video all four), 10.5 s each at real time plus a 2 s hold of the last frame. One panel: recorded humans (outlined, on top) and fine-tuned policy + BH-CBF (filled, no trails); a thin line joins the two versions of each vehicle, near-black where the policy is ahead or level, red where the human is ahead by more than 0.5 m along the recorded path (`progress`). The footer states which vehicles are ours (filled) and which are human (outlined). | `outputs/tits26/stage3_human_ft_recorded_maxlen6.5/natural/` (episode pickles, `matched_tracks.pkl`) | `HUMAN_SCENES`, `HUMAN_SEED` |
| 13 | `logos` | Closing card, 3 s: the logos of both universities, one above the other. | `assets/logo_*.png` | `scene_logos` |

Extra clips, not part of the video (`EXTRAS` in `video.py`; `render --only <name>` writes
`OUT/extra/<name>.mp4`, which `assemble` skips, and raw videos in `OUT/raw/`):

| Clip | Content | Source data | Key constants |
|---|---|---|---|
| `stage4_side_by_side` | The earlier Stage 4 layout: recorded humans left, policy with trails right (raw: `stage4_side_by_side_scene_{1..4}`; the project page shows scene 1). Labels with arrows point at the blue vehicle: "Slow down", then "Stop" for the human (left), and "Decisively enter" for the policy (right) from the human's slow-down until 1 s after the policy reaches the ring. | as `stage4` + `tails/s3_{a..d}.pkl` | `scene_humans(overlay=False)`, `HUMAN_CUES` (label times and position) |
| `repair` | Repair by lazy constraint generation, seen from vehicle i (heading up, roundabout). 3 s at real time, then one filter problem paused: the first QP's constraints near their bound (linearized at the nominal control), the candidate, the exact check with its violated (e, t), the added QP constraints, the second candidate, which passes; then 3 s at real time. Right: one cell per QP constraint (safety constraint e × prediction step t). | Stage 2 comparison run with BH-CBF (`COMPARISON_RUN/ours/dec/seed202`) + `repair/repair_round.pkl` | `REPAIR_EVENT` (step 147, vehicle 6), `REPAIR_LEAD`, `REPAIR_VIEW` |

Raw videos: `backup_round`, `backup_ind`, `stage1_rewards`, `stage2_counterfactual_{1..4}`,
`stage2_comparison`, `stage3_finetuning`, `stage4_scene_{1..4}`.

## Visual conventions

- **Colors:** the RWTH palette; key vehicles in blue, green, purple, and magenta, the others in gray.
  Collisions are drawn as a red outline plus a star.
- **Collision counter:** a collision of two vehicles counts once (lane collisions count once each).
- **Braking trails:** drawn wherever BH-CBF acts, except in Stage 4, as opaque light tints of the
  vehicle's color (`Tails`, `tint`), so overlapping footprints keep one shade. The standstill
  footprint is outlined.
- **Human drivers (Stage 4):** outlined, as in the paper. The outline lies inside the footprint, so
  outlined and filled vehicles look the same size. The joining lines are near-black by default and
  red only where the human is ahead.
- **Font:** Carlito (Calibri-compatible) throughout.

## How the scenes were chosen (and why)

- **Rewards (05):**
  - The window starts at t = 0; the three policies share the map, training seed, and deployment
    seed, so they start from the same joint state (verified: identical poses at step 0).
  - The BH-CBF-informed reward must have 0 collisions and the TTC and one-step CBF rewards at least
    1 each. `R select_windows rewards` lists the candidates (only roundabout ones qualify).
- **Comparison (08):**
  - The window must start at t = 0, so that all three filters start from the same state.
  - BH-CBF must have 0 collisions, the one-step filter at least 1, and no filter at least as many
    as the one-step filter.
  - `R select_windows comparison --steps 250` lists the candidates.
- **Fine-tuning (10):**
  - Seeds 201–205 were rolled out with `R rollouts … --seed <k>`; `R select_windows stage2` compares them.
  - The video shows seed 205.
  - The demo rollouts use 1 environment, unlike the paper's evaluation, so their collision counts
    are examples, not paper numbers.
- **Backup scene, intersection (03):** `R select_windows backup --map ind_location4 --steps 100`; this is the
  window with the most moving vehicles in view.
- **Counterfactuals (07) and Stage 4 (12):** the scenes of the paper's figures. Their parameters
  come from `paper_fig_stage1_counterfactual.py` and
  `paper_fig_stage3_footprints.py` (`SCENES`).

## Known subtleties

- **Respawns differ between rollouts.**
  - A vehicle that reaches its exit (or collides) respawns at a random entry. In the
    counterfactual scenes, the branch (no filter) and the master (BH-CBF) draw different respawn
    positions, so the panels would show different traffic.
  - `hide_after_respawn` therefore hides each vehicle from its first respawn on, in both panels.
    The footer says so.
  - Respawns are also why a rerun "misses" a few steps when capturing tails (the stored pose
    before the respawn doesn't match the state the filter reads). `tails_at` then holds the
    previous step's tail for 0.1 s, which isn't visible.
- **Tails must belong to the replayed poses.** `tails.py` captures a state only if its
  positions match the stored rollout (tolerance 1e-4 m) and prints the missed steps. Stage 4 reruns
  are verified against the paper's episode pickles (`deviation`).
- **Counterfactual rollouts end at step 1200.** So scene 4 can continue only 24 steps after its
  collision (`len(filtered) - 1` caps it).
- **Interpolation:** rollouts are at 10 Hz; frames interpolate poses linearly (heading wrapped),
  but never across a respawn jump (> 3 m).

## Recipes for common changes

- **Change text** (titles, subtitles, bullets, footer notes): edit the strings in the `scene_*`
  functions or in `CLIPS`, then re-render that clip and `assemble`.
- **Change the speed or length of a scene:** `speed=` of `scene_panels`/`scene_finetuning`/
  `scene_humans`, or the `*_STEPS` constants. For counterfactual scenes, `CONTINUE[<cf_x>]` sets
  the steps after the collision. **If a span grows past the captured tails, recapture them**
  (below); otherwise the last captured tail is held.
- **Use another scene or window:** change the constants; for BH-CBF panels, capture tails for the
  new run and steps:
  ```bash
  R tails deployment --run <Stage 2 ours/dec run folder> --steps <a> <b> --name <name>
  R tails counterfactual --npz <label>/<map>/seed301.npz --steps <branch> <end> --name cf_x
  R tails humans --episode <recName> --seed 1 --name s3_x
  ```
  Each takes a few minutes (it reruns the deployment up to the last step) and overwrites
  `OUT/tails/<name>.*`.
- **More Stage 3 seeds or checkpoints:**
  `R rollouts --run outputs/tits26/stage2_finetune/<label>/<map>/ours/seed1 --iter 0 100 --seed <k>`.
- **Window of a map panel:** `(x0, x1, y0, y1)` in meters. The panel keeps the aspect ratio, so a
  wide window makes the panels short (as in the three-panel comparison).
- **Cover page:** `scene_title` (positions in figure fractions, image widths in figure widths).
- **Add a scene:** write a `scene_<name>(clip)` function that builds a figure with `new_figure`
  and `panel_axes`, then calls `clip.add(fig)` per frame. Call `clip.start_raw(name, fig, axes)`
  before the frames and `clip.stop_raw()` after them for a raw video. Add it to `CLIPS`; the clip
  number is its position in that list.

## Explainer video (how the filter works)

A second, separate video (about 3 minutes) zooms into single filter problems of the
decentralized BH-CBF filter, in the order of the paper's method section:

- **Final video:** `OUT/bh_cbf_explainer.mp4`; clips in `OUT/explainer/clips/` (each one usable as a loop on the project page).
- **Scripts:** `explainer_capture.py` (data) and `explainer.py` (`render`, `assemble`).

`explainer_capture.py` reruns the Stage 2 comparison run with BH-CBF (the run of the `repair` clip) and
records, for the given steps and vehicles, every filter problem in full: the vehicle table and the
backup predictions, the shares and corner-budget data, the lane facets, the exact values of all
inequalities on a 61 × 61 grid over the admissible box of (a, ω), the predictions along the path from the
policy's control to the filtered one, and the QP constraints as half-planes. It checks that its own
evaluation reproduces the filter's values. Output: `OUT/explainer/<name>.pkl`.

```bash
RUN=outputs/tits26/stage1_filters_ttc_iter100/round_location0__stage1_ttc_seed_4__iter100/round_location0/ours/dec/seed202
R explainer_capture --run $RUN --steps 94 120 --agents 9 --name explainer_lane
R explainer_capture --run $RUN --steps 112 155 --agents 6 10 --name explainer_pair
R explainer render [--only <clip> ...]
R explainer assemble
```

| # | Clip | Content | Key constants |
|---|---|---|---|
| 00 | `intro` | Card. | `CLIPS` |
| 01 | `one_step` | A vehicle at 11.7 m/s approaches a stopped vehicle on a straight road; the one-step CBF filter becomes infeasible and collides, BH-CBF stops 0.1 m behind. Simulated in `explainer.py` (`simulate_line`) with the paper's model, margin, and parameters; 0.5× speed. | `simulate_line` |
| 02 | `backup` | Vehicle 9 at step 114 (exit lane, next to its left boundary): backup prediction, margin to the left boundary, backup value, the red line (1 − λ)B_e, the candidate of the policy's control (violates), and the sweep to the filtered control (steers less). | `LANE_*` |
| 03 | `control_plane` | The same problem in the (a, ω) plane: exact feasible set, policy's control, backup control, QP rows and candidate, filtered control with its QP-cost circle (axes scaled by the QP weights); then steps k+1 and k+2. | `ControlPlane` |
| 04 | `recursive` | The filtered candidate at k equals the backup prediction at k+1 (verified to 1e-5 m), so B_e(x_{k+1}) ≥ (1 − λ)B_e(x_k). | – |
| 05 | `pair` | Vehicle 10 (i) approaches vehicle 6 (j) at step 130: pair backup value, the budget split into shares (0.80 / 0.20), the corner budgets at the step where the policy's control of i violates most (zoom, separating normal, strips), the two control planes, then real time. | `PAIR_*` |
| 06 | `summary` | Four takeaways. | – |

Why these events: lane events inside the drawn map are rare in this run (most are artifacts at map
exits, where the route boundary extends beyond the drawn road); step 114 of vehicle 9 is the clearest
one with only a lane constraint active. Step 130 is a car-following conflict in the ring in which the
share rule (the approaching vehicle takes the larger share) is visible.
