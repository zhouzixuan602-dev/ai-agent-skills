#!/bin/bash
# ============================================================
# 首次运行：初始化 GitHub 仓库
# 运行方式: bash setup_github.sh
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "🚀 初始化 AI Agent Skills GitHub 仓库..."

# 检查配置
if [ "$GITHUB_USERNAME" = "YOUR_GITHUB_USERNAME" ] || [ "$GITHUB_TOKEN" = "YOUR_GITHUB_TOKEN" ]; then
    echo "❌ 请先编辑 config.sh，填入你的 GitHub 用户名和 Token"
    exit 1
fi

# 进入仓库目录
cd "$LOCAL_REPO_PATH"

# 初始化 git（如果还没有）
if [ ! -d ".git" ]; then
    git init
    git config user.name "$GIT_USER_NAME"
    git config user.email "$GIT_USER_EMAIL"
    echo "✅ Git 初始化完成"
fi

# 在 GitHub 上创建仓库（如果不存在）
echo "📦 在 GitHub 上创建仓库..."
HTTP_STATUS=$(curl -s -o /tmp/github_create_resp.json -w "%{http_code}" \
    -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/repos \
    -d "{
        \"name\": \"$REPO_NAME\",
        \"description\": \"$REPO_DESCRIPTION\",
        \"private\": false,
        \"auto_init\": false
    }")

if [ "$HTTP_STATUS" = "201" ]; then
    echo "✅ GitHub 仓库创建成功: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
elif [ "$HTTP_STATUS" = "422" ]; then
    echo "ℹ️  仓库已存在，继续配置..."
else
    echo "⚠️  仓库创建返回状态: $HTTP_STATUS"
    cat /tmp/github_create_resp.json
fi

# 设置远程地址
git remote remove origin 2>/dev/null || true
git remote add origin "https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "✅ 远程仓库配置完成"

# 首次推送
git add -A
git diff --cached --quiet || git commit -m "🚀 初始化 AI Agent Skills 仓库"
git branch -M main
git push -u origin main 2>&1 | grep -v "token"

echo ""
echo "🎉 设置完成！"
echo "📌 仓库地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo "⏰ 每天墨尔本时间 2am 将自动生成并发布新 skill"
