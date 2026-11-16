# User documentation infrastructure

The user documentation is a static site built with [Sphinx](https://www.sphinx-doc.org) from the sources in [`docs/user/`](../../user) and published to GitHub Pages at <https://gridtools.github.io/gt4py>.

## Site structure

The site is made of three **independent Sphinx projects**, each one with its own `conf.py`, which are built into a single output tree:

| Site              | Sources                                        | Published at  | Source format                 |
| ----------------- | ---------------------------------------------- | ------------- | ----------------------------- |
| Welcome page      | [`docs/user/`](../../user)                     | `/`           | MyST Markdown                 |
| `gt4py.cartesian` | [`docs/user/cartesian/`](../../user/cartesian) | `/cartesian/` | reStructuredText              |
| `gt4py.next`      | [`docs/user/next/`](../../user/next)           | `/next/`      | MyST Markdown, MyST notebooks |

Keeping them as separate projects means that each subsite has its own navigation, search and API reference, and can be browsed without the other one. The welcome page only introduces the project and links to both subsites.

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

New pages should be written in [MyST Markdown](https://myst-parser.readthedocs.io). The existing reStructuredText pages of the `cartesian` subsite are kept as they are.

### Executable pages (MyST notebooks)

In the `next` subsite, Markdown files with a `jupytext` header declaring the `myst` format are notebooks: [MyST-NB](https://myst-nb.readthedocs.io) executes their `{code-cell}` blocks when the site is built and renders the outputs. A failing cell fails the build, so the examples in the documentation cannot silently go out of date. `QuickstartGuide.md` is an example.

Markdown files without that header are rendered as they are, without running their code blocks.

The same files are also run as tests by the `test_examples` nox session, which converts them to `.ipynb` notebooks with `jupytext` and executes them with `nbmake`. The generated `.ipynb` files are excluded from the Sphinx sources.

Prefer executable pages for anything showing GT4Py code.

### Layout

- Grids, cards and tabs are available through [`sphinx-design`](https://sphinx-design.readthedocs.io).

## API references

- `gt4py.cartesian`: generated for all modules with `sphinx-apidoc` at build time into `docs/user/cartesian/_source/` (not tracked by git). It covers every subpackage except `gt4py.next` and `gt4py.eve`.
- `gt4py.next`: a **curated** page, [`docs/user/next/apiref.md`](../../user/next/apiref.md), which documents only the public API, that is, the names exported in `gt4py.next.__all__`. When a new name is exported there, add it to this page.

Docstrings are written in Google style (see the [coding guidelines](../../../CODING_GUIDELINES.md)) and rendered with `sphinx.ext.napoleon`.

## Publication

The [`Documentation`](../../../.github/workflows/docs.yml) GitHub workflow runs the `docs` nox session:

- On pull requests changing the documentation, the sources or the build configuration, it only builds the site and uploads it as a workflow artifact (`github-pages`), which can be downloaded to review the result.
- On pushes to `main` of the upstream repository, it additionally deploys that artifact to GitHub Pages.

This requires the Pages source of the repository to be set to _GitHub Actions_ (in _Settings > Pages_). Every deployment replaces the whole published site with the result of the build: nothing is stored in a git branch, and only the documentation of the current `main` is available.

The `latest/` path, which older links to the documentation use, redirects to the welcome page.
