# Research Website V1

Astro static website for the GeoAI lightning talk research page.

## Preview Locally

From this directory:

```bash
npm install
npm run dev
```

Then open:

```text
http://localhost:4321
```

Build the production site with:

```bash
npm run build
```

Preview the production build with:

```bash
npm run preview
```

## Publish Options

For the cleanest GitHub Pages setup, create a dedicated repo and put the contents of this `website/` directory at the repo root.

For this existing repo, either:

- Use the included `.github/workflows/deploy-website.yml` workflow to build and deploy this `website/` directory.

For the included workflow, enable GitHub Pages in the repository settings and choose "GitHub Actions" as the source.

## Structure

- `astro.config.mjs`: Astro static-site config.
- `package.json`: local dev/build scripts and Astro dependency.
- `src/layouts/BaseLayout.astro`: shared page shell, navigation, MathJax, footer.
- `src/pages/index.astro`: research content.
- `src/styles/global.css`: black, white, and grey visual system.
- `public/assets/`: copied plots and downloadable result artifacts.

## Update After Final Aurora Adaptation Run

Replace the Aurora CSVs and plots in `public/assets/data/` and `public/assets/`, then update the WeatherBench2 result cards and table in `src/pages/index.astro`.
