# Branches and push order

The original `main` branch already has the Plain RAG baseline. The following
local branches keep each experiment change reviewable:

| Order | Branch | Review against | Change |
|---|---|---|---|
| 1 | `codex/ramdocs-72b-comparison` | `main` | Shared A/B runner, identical final model/settings/scoring, paired outputs, Sol batch setup |
| 2 | `codex/ramdocs-72b-card-extraction` | `codex/ramdocs-72b-comparison` | C: fresh 72B cards using the existing card recipe, resumable document records, C evaluation |
| 3 | `codex/ramdocs-72b-card-review` | `codex/ramdocs-72b-card-extraction` | D: question-fit checking, unchanged retained cards, combined A–D report and full Sol workflow |

These branches are stacked: C contains A/B, and D contains C. The existing
`codex/ramdocs-repro-package` branch is unchanged. No experimental GPU results
are included in the code commits.

The next branch to push is the A/B comparison branch. These are manual commands;
preparing this document does not push them:

```bash
git push -u origin codex/ramdocs-72b-comparison
```

After reviewing A/B, publish C and D in order:

```bash
git push -u origin codex/ramdocs-72b-card-extraction
git push -u origin codex/ramdocs-72b-card-review
```

When opening pull requests before the earlier changes are merged, use the
"Review against" branch as the base so the review shows only the next version.
After all branches are published, use the D checkout on Sol to run the complete
workflow. The submission wrapper snapshots code and inputs before queuing, so
later checkouts do not alter submitted jobs. Start with a five-question smoke
run before requesting all 500 questions.
