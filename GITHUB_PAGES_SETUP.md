# GitHub Pages Setup for `CoRAL-ASU/diagrams`

This workspace now contains a deploy workflow and starter site:

- `.github/workflows/deploy-pages.yml`
- `pages/index.html`
- `pages/styles.css`

## 1. Copy into your `diagrams` repository

From your local clone of `https://github.com/CoRAL-ASU/diagrams`, copy these files into the same paths.

## 2. Commit and push

```bash
git add .github/workflows/deploy-pages.yml pages
git commit -m "Add GitHub Pages site and deployment workflow"
git push origin main
```

## 3. Enable GitHub Pages (one time)

In GitHub:

1. Open `Settings` -> `Pages`
2. Under `Build and deployment`, select `Source: GitHub Actions`

## 4. Your site URL

After the workflow finishes, your site should be available at:

`https://coral-asu.github.io/diagrams/`
