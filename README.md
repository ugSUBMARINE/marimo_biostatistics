# Biostatistics notebooks

Interactive [marimo](https://marimo.io/) notebooks accompanying the Biostatistics lecture course (MOL.824UB at the University of Graz). Explore descriptive and inferential statistics, correlation and regression, hypothesis testing, and Bayesian statistics through examples, simulations, and interactive plots.

## Read online

Open the [course website](https://ugsubmarine.github.io/marimo_biostatistics/) to explore all six chapters in your browser. No local Python installation is needed; the first load may take a little time.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone or download this repository, and run these commands from its folder:

```sh
uv sync --locked
uv run marimo run 01_descriptive_statistics.py
```

`uv` manages Python (3.14+) and the required dependencies. The notebook opens in your browser. Replace the filename with any of the numbered notebooks (`01`–`06`), or use `uv run marimo edit` to browse and edit them.

To open a [gallery](https://docs.marimo.io/guides/apps/#gallery) with clickable cards for all six notebooks:

```sh
uv run marimo run 0*.py
```

## Build and preview the website

The [build script](scripts/build-site.sh) exports all six notebooks as interactive
WebAssembly pages and combines them with the landing page:

```sh
bash scripts/build-site.sh
uv run python -m http.server --directory _site 8000
```

Open [localhost:8000](http://localhost:8000). Stop the server with Ctrl+C.
The exported notebooks must be served over HTTP rather than opened directly as
files.

### Source files and generated output

- `01_*.py`–`06_*.py`: notebook sources.
- `public/`: CSV datasets used by the notebooks.
- `companion_*.py`: shared data-loading, statistics, and presentation helpers.
- `site/`: landing-page HTML and the University of Graz logo.
- `_site/`: generated website, including notebook assets, datasets, and packaged
  helper modules. This directory is excluded from Git.

Use `mo.notebook_location()` with `companion_data.read_csv()` to load course
CSVs consistently in native Python and WebAssembly.

The build script accepts an optional output directory and overwrites existing
exports without removing older files. Use a new directory for a clean preview.

## Publishing updates

The [GitHub Pages workflow](.github/workflows/pages.yml) builds and publishes the
website on every push to `main`. It uses `.python-version` and `uv.lock` for the
build environment and checks CSV loading before exporting the notebooks.

To update the website, edit the source files, commit, and push to `main`.
Follow deployment progress in the repository's
[Actions tab](https://github.com/ugSUBMARINE/marimo_biostatistics/actions).
The **Deploy course website** workflow can also be run manually on `main`.
After deployment, check the affected chapters and their controls in the browser.

Keep dependency changes in `pyproject.toml` and `uv.lock` together. Commit source
files rather than the generated `_site/` directory.

## Acknowledgements

Development has included assistance from OpenAI's Codex.
