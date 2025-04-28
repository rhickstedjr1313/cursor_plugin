#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────
GH_USER="rhickstedjr1313"
REPO="cursor_plugin"
VISIBILITY="private"        # use “private” if you want a private repo
# You must export a GitHub token with repo‑creation rights:
export GITHUB_TOKEN="<Your Token>"
# ──────────────────────────────────────────────────────────────────────

if [ -z "${GITHUB_TOKEN-}" ]; then
  echo "❌ Please set GITHUB_TOKEN (a GitHub PAT with repo scope)."
  exit 1
fi

# 1) Create the GitHub repo via REST API
echo "→ Creating GitHub repo $GH_USER/$REPO…"
curl -sS -H "Authorization: token $GITHUB_TOKEN" \
     -d "{\"name\":\"$REPO\",\"private\":$( [ "$VISIBILITY" = "private" ] && echo true || echo false )}" \
     https://api.github.com/user/repos \
  | jq .html_url >/dev/null || { echo "❌ Repo creation failed"; exit 1; }

# 2) Initialize local git if needed, commit
[ -d .git ] || git init
git add .
git commit -m "Initial commit"

# 3) Push to GitHub
REMOTE="https://github.com/$GH_USER/$REPO.git"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git push -u origin HEAD:main

echo "✅ Done! Repo is https://github.com/$GH_USER/$REPO"
