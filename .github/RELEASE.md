# Release Process

## Creating a Release

```bash
python scripts/bump_version.py patch  # or: minor, major
git push && git push --tags
```

Automation handles the rest: https://github.com/parameterlab/maseval/actions

---

## Troubleshooting

**Wrong version tagged?**

```bash
git tag -d v0.1.2
git push origin :refs/tags/v0.1.2
# Then fix pyproject.toml and retry
```

**Automation failed?**

```bash
uv build && uv publish
```

## Version Guide

- `patch`: Bug fixes (0.1.1 → 0.1.2)
- `minor`: New features (0.1.1 → 0.2.0)
- `major`: Breaking changes (0.1.1 → 1.0.0)
