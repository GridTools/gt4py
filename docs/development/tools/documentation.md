# User documentation infrastructure

The user documentation is a static site built with [Sphinx](https://www.sphinx-doc.org) from the sources in [`docs/user/`](../../user) and published to GitHub Pages at <https://gridtools.github.io/gt4py>.

## Site structure

The site is made of three **independent Sphinx projects**, each one with its own `conf.py`, which are built into a single output tree:

| Site              | Sources                                        | Published at  | Source format                 |
| ----------------- | ---------------------------------------------- | ------------- | ----------------------------- |
| Welcome page      | [`docs/user/`](../../user)                     | `/`           | MyST Markdown                 |
| `gt4py.cartesian` | [`docs/user/cartesian/`](../../user/cartesian) | `/cartesian/` | reStructuredText              |
| `gt4py.next`      | [`docs/user/next/`](../../user/next)           | `/next/`      | MyST Markdown, MyST notebooks |

Keeping them as separate projects means that each subsite has its own navigation and search, and can be browsed without the other one. The welcome page only introduces the project and links to both subsites.

What the three projects share lives directly in `docs/user/`:

- `site_theme.py`: the theme settings (theme, colors, logo, sidebar), imported by every `conf.py`.
- `_templates/`: template components, like the one rendering the links to the other sites.
- `_static/`: stylesheets.

Links between the sites are always **relative** (`../next/index.html`), never absolute, because the site is served from a subfolder (`/gt4py/`). In the templates this is done with the `pathto()` Sphinx helper. Links to the other sites are declared as the `site_links` argument of `theme_settings()` in each `conf.py`.

## Building the site

The `docs` nox session builds the three projects in a dedicated environment:

```bash
./noxfile.py -s docs  # output in 'docs/user/_build/html'
```

Arguments after `--` are forwarded to all the `sphinx-build` invocations. The result can be checked locally with any static file server, for example `python -m http.server -d docs/user/_build/html`.

A single project can also be built directly from the development environment, which is faster while writing:

```bash
uv run sphinx-build -b html docs/user/next docs/user/_build/html/next
```

The tools are declared in the `docs` dependency group of `pyproject.toml`.

### Sphinx and Python versions

Recent Sphinx releases drop Python versions faster than GT4Py does, so the Sphinx requirement in the `docs` group is split by Python version. The `docs` nox session uses the newest supported Python version, which is therefore the one defining the Sphinx version of the published site. A development environment using an older Python version may get an older Sphinx.

## Writing pages

New pages should be written in [MyST Markdown](https://myst-parser.readthedocs.io). The existing reStructuredText pages of the `gt4py.cartesian` subsite are kept as they are.

### Executable pages (MyST notebooks)

In the `gt4py.next` subsite, Markdown files with a `jupytext` header declaring the `myst` format are notebooks: [MyST-NB](https://myst-nb.readthedocs.io) executes their `{code-cell}` blocks when the site is built and renders the outputs. A failing cell fails the build, so the examples in the documentation cannot silently go out of date. `quickstart.md` is an example.

Markdown files without that header are rendered as they are, without running their code blocks.

The same files are also run as tests by the `test_examples` nox session, which converts them to `.ipynb` notebooks with `jupytext` and executes them with `nbmake`. The generated `.ipynb` files are excluded from the Sphinx sources.

The pages of the `gt4py.next` subsite are organized in folders: `dsl/` (what can be written inside operators), `python/` (the Python side: grids, fields and execution) and `examples/` (complete small applications). The `test_examples` session picks up every executable page in these folders, so a new page only has to be added to a `toctree`. A new folder must also be added to the list in that session.

Prefer executable pages for anything showing GT4Py code.

### Layout

- Grids, cards and tabs are available through [`sphinx-design`](https://sphinx-design.readthedocs.io).

## API reference

Only the `gt4py.cartesian` subsite has an API reference. It is generated for all modules with `sphinx-apidoc` at build time into `docs/user/cartesian/_source/` (not tracked by git), and it covers every subpackage except `gt4py.next` and `gt4py.eve`. Docstrings are written in Google style (see the [coding guidelines](../../../CODING_GUIDELINES.md)) and rendered with `sphinx.ext.napoleon`.

The `gt4py.next` subsite does not include an API reference generated from docstrings.

## Publication

The [`Documentation`](../../../.github/workflows/docs.yml) GitHub workflow runs the `docs` nox session:

- On every pull request, it only builds the site and uploads it as a workflow artifact (`github-pages`), which can be downloaded to review the result. It runs with no path filter, so that its `Build documentation` check can be a required status check of the `main` branch: a broken documentation build then blocks merging.
- When a GitHub release is published (prereleases are ignored), it builds the site from the release tag and deploys it to GitHub Pages. This is the only automatic publication: merging to `main` does not change the published site.
- As a manual escape hatch, for example to fix the documentation of the latest release without making a new one, the workflow can be run from the _Actions_ tab selecting a release **tag** (not a branch) and enabling the `deploy` input. On any other ref the deployment is skipped.

Deploying requires:

- the Pages source of the repository set to _GitHub Actions_ (in _Settings > Pages_);
- the `github-pages` environment allowing deployments from the release tags (in _Settings > Environments_, add the `v*` tag pattern to the deployment branches and tags).

Every deployment replaces the whole published site with the result of the build: nothing is stored in a git branch, and only the documentation of the last published release is available.

The `latest/` path, which older links to the documentation use, redirects to the welcome page.
