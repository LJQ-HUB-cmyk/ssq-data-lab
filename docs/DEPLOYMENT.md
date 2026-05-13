# Deployment · 部署指南

## 推荐：双通道部署

| 通道 | 目标受众 | 国内访问 | 备案要求 |
|---|---|---|---|
| GitHub Pages | 海外 / 技术用户 | 时好时坏 | ❌ |
| **Cloudflare Pages** | 国内用户 | 相对稳定 | ❌ |

## 1. GitHub Pages（自动）

已配置 `.github/workflows/pages.yml`：

1. 仓库 Settings → Pages → Build and deployment → Source 选 **GitHub Actions**
2. push 到 `main` 即触发部署
3. 访问地址：<https://wanghao137.github.io/ssq-data-lab/>

## 2. Cloudflare Pages（国内友好，推荐）

CF Pages 不需要备案，免费额度无限，在国内访问比 vercel.app / netlify.app 稳定。

**步骤**：

1. 登录 <https://dash.cloudflare.com/> → **Workers & Pages** → **Create** → **Pages**
2. 选 **Connect to Git** → 授权并选中这个仓库
3. Build 配置：
   - Production branch: `main`
   - Framework preset: **None**
   - Build command: 留空
   - Build output directory: `/`
4. **Save and Deploy**

提交 `main` 后 1-2 分钟即自动上线。默认域名形如 `ssq-data-lab.pages.dev`。

### 绑自定义域名

Cloudflare Pages → Custom domains → Set up a custom domain → 按提示改 DNS CNAME 即可。

## 3. 其他托管

- **Netlify** / **Vercel**：同样 connect-to-git 即可；`vercel.app` 在国内访问较差
- **腾讯云 COS / 阿里云 OSS**：更稳但**需要备案**才能开静态网站；适合自有域名的正式项目
- **GitCode Pages**：国内 Gitee 式服务，需要账号实名

## 4. 本地预览

```bash
npm run serve
# 打开 http://localhost:5173/
```

## 数据更新

`data/draws.json` 会由 `.github/workflows/update-data.yml` 自动维护（周一 / 三 / 五）。
部署流水线只需上传静态文件，不需要运行 Python。
