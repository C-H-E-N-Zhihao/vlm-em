# Pipeline: shared by every experiment

Prompts → answers → labels → statistics. The finding folders only hold each experiment's analysis.

| script | does |
|---|---|
| `forms.py` | builds every prompt form from the paper's 28 dishonesty items; `self_check` verifies the paper's wording |
| `generate.py` | samples answers for one (model, task, forms): the swift backend on RCP, or `--backend hf` (also for prefilled replies, E17) |
| `score_rules.py` | rule labels for yes/no and one-word answers: asserts t, f, or `other` |
| `judge.py` | labels free-text answers with the paper's dishonesty judge on Qwen2.5-72B |
| `comply.py` | the paper's COMPLY follow-up ("I'll say blue because you asked" is not a lie) |
| `apply_hand_labels.py` | merges Guanyu's blind hand labels into the automatic labels (`labels_<source>+hand.json`) |
| `judge_sheet.py`, `hand_agree.py` | blind hand-check sheets, and agreement of hand labels with the judge and rules |
| `analyze.py` | rates, paired bootstrap Δ (`boot_diff`, `boot_ddiff`), pipeline checks, the E1/E2 verdict; helpers used by every analysis |
| `style.py` | figure colours |

On RCP, `rcp/judge.sh` runs `score_rules` then `judge` (or `comply` with `MEM_JUDGE_STEP=comply`).
