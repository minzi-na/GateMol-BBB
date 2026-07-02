# Final model architectures

One file per protocol, extracted verbatim from the **final** milestone of each
AutoResearch run (only the `nn.Module` definitions; the training harness is
reimplemented cleanly in `gatemol_bbb.train`).

| Module | Protocol | Source commit | Milestone |
|---|---|---|---|
| `protocol1_gmlp.py` | I  | `88fdf45` (`bbb_train.py`) | iter90 attention-pooling arch + Phase-2 HPO |
| `protocol2_gmlp.py` | II | `2e8cbdc` (`train.py`) | iter198 single-head attn-biased pool (Phase-2 did not beat it) |

Scope: final models only. Intermediate architecture-search milestones are not
part of this release.
