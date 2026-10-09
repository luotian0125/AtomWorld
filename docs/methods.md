# Methods and metrics

## Kinetic calibration

For physical event rates Γₐ(s), the generator is
`L f(s) = Σₐ Γₐ(s)[f(s′ₐ) − f(s)]`.
The mean first-passage time to a specified absorbing target satisfies
`L τ = −1`, with `τ = 0` on the target. The finite-state solver checks target
reachability and computes the first and second moments.

`τ(s) − τ(s′)` is signed potential progress, not a physical time increment.
The BCC residence model instead predicts the positive conditional mean time to
the next physical event: `t_hold(s) = 1 / Σₐ Γₐ(s)`.
It fits three Arrhenius parameters from species indicators and `ΔE/kBT`, sums
over the complete legal event catalogue and uses independent passive-KMC data.
Known event rates and actor features are excluded from model inputs.

Local residence calibration does not restore the original physical process when
the policy changes event probabilities. Physical-time dynamics errors for biased
policies are therefore reported as N/A.

## BCC metrics

All reference comparisons use paired initial states and matched event budgets.
Uncertainty across evaluation seeds is distinct from variability across
independently trained policies.

| Metric | Definition |
|---|---|
| Energy AUC | Integral of energy decrease over normalized event progress; method ensemble mean divided by the KMC ensemble mean. Uncertainty uses paired bootstrap. |
| ClockErr_hold | Per-trajectory mean `abs(log(predicted mean residence / exact mean residence))`, averaged across seeds. |
| StructErr | Progress integral of the mean absolute difference in clustered-Cu fraction and largest-cluster fraction against paired KMC. |
| Arrival hit | Fraction of vacancies ever in 1NN contact with a connected Cu cluster of size ≥2, including initial contact. Size ≥4 is also stored. |
| AF | KMC total event budget divided by the method's first event index with energy ≤ paired KMC final energy. |
| First-passage step Speedup | KMC first-hit event index divided by method first-hit index for the same target. |
| Restricted-mean step ratio | Ratio of mean first-hit costs truncated at the evaluation budget; includes all trials. |
| Fixed-budget wall Speedup | Ratio of ensemble mean recorded evaluation times for the same event budget. |

AF and first-passage Speedup means use finite successful pairs only and must be
read with the reached/total count. Unreached targets remain right-censored;
initially satisfied targets have undefined AF. A single finite hit has no SE.
The truncated-cost ratio is not an uncensored first-passage estimate.

Wall timing includes simulation, diagnostics, clock inference and NPZ output;
it excludes training, initialization and scheduler waiting. Per-event wall times
were not recorded, so wall-time-to-target Speedup is unavailable.

## Scope

BCC higher-neighbour pair parameters include illustrative values. These results
describe the specified numerical model, not validated Fe–Cu material kinetics.
Residence checkpoints are restricted to their recorded physical domain.
FCC/HCP parameter sources are recorded in their configurations.
