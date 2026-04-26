# Jimsky ACE-Step Fork Strategy

This fork is the Jimsky / The Mind Expansion Network working copy of ACE-Step 1.5.

- Fork: `https://github.com/TheMindExpansionNetwork/ACE-Step-1.5`
- Upstream: `https://github.com/ace-step/ACE-Step-1.5`
- Local path: `/opt/data/workspace/projects/ACE-Step-1.5`
- License inherited from upstream: MIT

## Why fork instead of copying

Forking is the right move for this kind of portfolio/build-machine work because it keeps the chain of credit and history clean:

1. The original creators keep attribution.
2. Jimsky can build custom workflows, deployment layers, agent notes, UI experiments, and portfolio demos on top.
3. Upstream improvements can still be pulled in.
4. If Jimsky fixes something useful, we can open a clean upstream pull request later.

This is not “stealing stuff” when the license allows use and we preserve attribution. The professional path is: fork, credit, document, build original layers, and avoid pretending upstream work is ours.

## Remote layout

```bash
origin   https://github.com/TheMindExpansionNetwork/ACE-Step-1.5.git
upstream https://github.com/ace-step/ACE-Step-1.5.git
```

## Recommended branch model

Best long-term model:

- `main`: Jimsky working branch for our deployed machine and portfolio experiments.
- `upstream/main`: original ACE-Step source of truth.
- `jimsky/*`: feature branches for our changes before merging into `main`.
- `vendor-sync/*`: temporary branches for syncing upstream updates and resolving conflicts safely.

Alternative stricter model if the fork gets messy later:

- Keep `main` as a clean mirror of upstream.
- Move all Jimsky work to `jimsky/main`.

For now, because this fork already has Jimsky-specific work, keep using `main` as the Jimsky branch and sync upstream carefully.

## Sync upstream safely

Do not blindly reset `main`; that would delete Jimsky work.

Preferred sync flow:

```bash
git fetch upstream --prune
git checkout main
git pull origin main
git checkout -b vendor-sync/upstream-$(date +%Y%m%d)
git merge upstream/main
# resolve conflicts, run tests/smoke checks
git push -u origin HEAD
```

After review, merge `vendor-sync/...` into `main`.

If the fork is very far behind upstream, use a dedicated sync branch and review with agents before merging.

## Agent rules for this fork

1. Preserve upstream attribution, license, and README references.
2. Keep Jimsky additions clearly labeled as Jimsky layers, wrappers, deployment scripts, docs, or experiments.
3. Do not commit model weights, generated audio, private prompts, API keys, `.env`, SSH keys, or paid credentials.
4. Prefer small, reversible commits.
5. Add tests or smoke checks for any code path that changes inference, API behavior, UI behavior, or deployment behavior.
6. For GPU/Modal/VPS work, default to scale-to-zero and include a shutdown/cleanup verification note.
7. Before major upstream merges, snapshot current `main`, fetch upstream, and produce a conflict/risk report.

## First Jimsky goals

- Get the repo downloaded and runnable in a bounded environment.
- Identify the cheapest useful local/Modal/VPS path for music generation experiments.
- Add a Jimsky wrapper/workbench layer without breaking upstream internals.
- Build portfolio demos that credit ACE-Step and show Jimsky orchestration, deployment, UI, and creative workflow improvements.
