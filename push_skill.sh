#!/bin/bash
# ============================================================
# 将新生成的 skill 推送到 GitHub
# 由定时任务在 Claude 生成 skill 后调用
# 用法: bash push_skill.sh <skill_folder_name> "<commit_message>"
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

SKILL_FOLDER="${1:-}"
COMMIT_MSG="${2:-"feat: add new AI agent skill"}"

if [ -z "$SKILL_FOLDER" ]; then
    echo "用法: bash push_skill.sh <skill_folder_name> '<commit_message>'"
    exit 1
fi

cd "$LOCAL_REPO_PATH"

# 确保 git remote 正确配置
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"
git remote set-url origin "https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# 拉取最新（避免冲突）
git pull origin main --rebase 2>&1 | grep -v "token" || true

# 暂存新 skill
git add "skills/$SKILL_FOLDER/"
git add "README.md" 2>/dev/null || true
git add "INDEX.md" 2>/dev/null || true

# 提交
git commit -m "$COMMIT_MSG" || echo "ℹ️  无新变更需要提交"

# 推送（隐藏 token 输出）
git push origin main 2>&1 | grep -v "token"

echo "✅ Skill 已推送到 GitHub: https://github.com/$GITHUB_USERNAME/$REPO_NAME/tree/main/skills/$SKILL_FOLDER"
