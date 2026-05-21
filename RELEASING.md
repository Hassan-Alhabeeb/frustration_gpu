# Releasing a new version

Step-by-step for cutting a new release. Future-you, follow this exactly. Replace `0.1.2` with whatever version you're cutting.

## 0. Pre-release sanity

```bash
cd F:/research_plan/frustration_gpu
git pull
python -m pytest tests/ --tb=no -q
```

223+ tests pass on local. If anything's red, fix that first.

## 1. Bump the version in two places

- `pyproject.toml` line ~7: `version = "0.1.2"`
- `CITATION.cff` line ~4: `version: 0.1.2` and `date-released: YYYY-MM-DD`

Both must match. CI doesn't check this, so be careful.

## 2. Write a CHANGELOG entry

Top of `CHANGELOG.md`, under `## [Unreleased]`, insert:

```markdown
## [0.1.2] - 2026-MM-DD

### Added / Changed / Fixed / Internal
- one bullet per change
```

Use the Keep-a-Changelog sections (Added, Changed, Deprecated, Removed, Fixed, Security). "Internal" is fine for workflow/build/CI fixes that don't affect users.

## 3. Commit + tag + push

```bash
git add pyproject.toml CITATION.cff CHANGELOG.md
git commit -m "v0.1.2: <one-line summary>"
git tag -a v0.1.2 -m "v0.1.2 — <one-line summary>"
git push origin main v0.1.2
```

Pushing the tag triggers the `publish.yml` workflow. The build job runs (~30 s), then the publish job pauses waiting for your approval.

## 4. Approve the PyPI deploy

Go to: https://github.com/Hassan-Alhabeeb/frustration_gpu/actions/workflows/publish.yml

Click the run for `v0.1.2`. You'll see a yellow "Review pending deployments" banner. Click it, check the `pypi` box, click **Approve and deploy**.

Workflow runs ~30 s, uploads the wheel + sdist to PyPI via Trusted Publishing (no token to manage).

Verify:

```bash
pip index versions frustration-gpu
```

Should show 0.1.2 as available.

## 5. Create the GitHub Release (for Zenodo DOI)

```bash
gh release create v0.1.2 --title "v0.1.2 — <one-line>" --notes-file - <<'EOF'
<release notes here, can paste from CHANGELOG.md>

## Install
pip install --upgrade frustration-gpu

## Full changelog
See CHANGELOG.md.
EOF
```

Or use the web UI: https://github.com/Hassan-Alhabeeb/frustration_gpu/releases/new

Creating the release fires the `publish.yml` workflow a second time. PyPI already has v0.1.2 so the upload step is a no-op (skip-existing is on). The green checkmark just means Zenodo got the webhook.

## 6. Verify Zenodo minted a DOI

Wait ~1 min, then:

```bash
curl -s 'https://zenodo.org/api/records?q=frustration_gpu&size=5' | python -c "import sys,json; [print(h.get('doi'), h.get('metadata',{}).get('version')) for h in json.load(sys.stdin)['hits']['hits']]"
```

You should see the new DOI for v0.1.2 plus the concept DOI (always `10.5281/zenodo.20323294`).

## 7. (Optional) Update the README DOI

The concept DOI never changes, so the README badge stays valid forever. Only update if you want to highlight a specific version DOI.

---

# Notes for special cases

## A workflow run shows red but PyPI is fine

Most likely a duplicate trigger (tag-push + release-published both fire `publish.yml`). The first one publishes, the second sees "version exists" and exits clean. `skip-existing: true` is set so this should NOT happen anymore — but if it does, ignore it, PyPI is the source of truth.

## I made a mistake in v0.1.2 and want to re-upload

You can't. PyPI versions are permanent. Bump to v0.1.3 and re-release.

If the wheel itself is broken (e.g. missing files), you can `yank` v0.1.2 on PyPI: https://pypi.org/project/frustration-gpu/0.1.2/ → admin panel → Yank. Yanking removes it from default `pip install` but doesn't delete the historical record.

## The publish workflow won't trigger

- Check the tag was actually pushed: `git ls-remote --tags origin | grep v0.1.2`
- Check the workflow file exists: `.github/workflows/publish.yml`
- Check the GitHub `pypi` environment still exists in repo Settings → Environments

## I'm releasing v1.0.0 and want to mark it major

Same flow. The version-bump rule is SemVer:
- patch (`v0.1.X`) — bug fixes, docs, no API change
- minor (`v0.X.0`) — new features, backward-compatible API
- major (`vX.0.0`) — breaking API changes

When you cut a major, write a "Migration from v0.X to v1.0" section in the release notes.
