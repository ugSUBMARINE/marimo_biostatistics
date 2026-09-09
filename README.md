# Biostatistics notebooks

Interactive [marimo](https://marimo.io/) notebooks accompanying the Biostatistics lecture course (MOL.824UB at the University of Graz). Explore descriptive and inferential statistics, correlation and regression, hypothesis testing, and Bayesian statistics through examples, simulations, and interactive plots.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone or download this repository, and run these commands from its folder:

```sh
uv sync
uv run marimo run 01_descriptive_statistics.py
```

`uv` manages Python (3.14+) and the required dependencies. The notebook opens in your browser. Replace the filename with any of the numbered notebooks (`01`–`06`), or use `uv run marimo edit` to browse and edit them.

To open a [gallery](https://docs.marimo.io/guides/apps/#gallery) with clickable cards for all six notebooks:

```sh
uv run marimo run 0*.py
```

## Acknowledgements

Development has included assistance from OpenAI's Codex.
