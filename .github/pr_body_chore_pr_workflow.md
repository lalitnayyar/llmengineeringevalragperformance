## Summary
- Add `.gitattributes` to normalize line endings across platforms.
- Keep key lock/config files stable with LF endings to reduce noisy diffs.

## Why
- Prevent CRLF/LF churn in PRs and reduce merge conflicts across Windows/macOS/Linux.

## Test plan
- [x] Repo still installs/runs normally (no runtime code changes).
- [x] Verify lock/docs no longer flip line endings unexpectedly.

