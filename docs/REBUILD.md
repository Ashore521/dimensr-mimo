# Repository reconstruction record

On 2026-10-25 (Asia/Shanghai), the owner directly authorized irreversible removal of the old local `dimensr-mimo` repository, experiments, environments, outputs and Git history, and replacement of the same GitHub repository's old content/history. The first deletion request was blocked by automatic approval review; no deletion ran then. After the owner's direct confirmation in this chat, the exact absolute target was verified, the clean checkout and absence of reparse points were checked, and only that directory was removed and recreated. No archive, backup branch or old tag was retained.

New local root: `0d117005bc121d4abf8a558ede361b8a50bd6c47`. It has no parent and no ancestry from the old MIMO repository.

The original DimenSR repository remains at `7fbcefadd105fe6266c8149cc05420de8ca74a76` with a clean worktree. Its environment and the project-root historical experiments/private documents were outside the deletion target and were not changed.

Remote reset used an atomic push with exact expected old branch/tag SHAs. Old `main` was replaced and old `v0.1.0` deleted. Before reset there were no releases or pull-request refs. Seventeen inventoried old Actions runs (and their associated artifacts) were deleted through GitHub's API. Network interruptions were resumed from per-record confirmations, without restarting the scientific experiments. `REMOTE_RESET.json` records the verification; only the new main ref and new successful CI run remained at that checkpoint.

This verifies removal of the old reachable branch/tag history and exposed workflow records in the named repository. It does not promise erasure of GitHub server caches/unreachable Git objects or copies previously downloaded by others. New v1 code, source attribution, tests and experiment evidence form a separate history.
