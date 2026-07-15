# Contributing to PyBIDS

Thanks for your interest in contributing to PyBIDS!

PyBIDS is part of the [BIDS](https://bids.neuroimaging.io/) ecosystem. For
general guidance on contributing to BIDS projects — using Git and GitHub, our
Code of Conduct, issue labels, and how contributions are recognized — please see
the [BIDS contributing guidelines][bids-contributing]. This document covers the
parts specific to PyBIDS: setting up a development environment, running the
tests, code style, and how we handle commits, pull requests, and the changelog.

## Where to ask questions

- **Bug reports, feature requests, and technical discussion:**
  [open an issue](https://github.com/bids-standard/pybids/issues).
- **Usage questions** (where no bug is suspected): the
  [NeuroStars](https://neurostars.org/tags/pybids) forum, under the `pybids` tag.

If you are unsure whether a change is wanted, open an issue to discuss it before
investing significant effort.

## Setting up a development environment

1. [Fork](https://github.com/bids-standard/pybids/fork) the repository and clone
   your fork.
2. PyBIDS uses a Git submodule for test data. After cloning, initialize it:

   ```bash
   git submodule update --init
   ```

   This checks out
   [`bids-examples`](https://github.com/bids-standard/bids-examples), which
   several tests rely on.
3. Testing is managed with [tox](https://tox.wiki/) (via `tox-uv`), which builds
   isolated environments so the same commands run locally and in CI:

   ```bash
   pip install tox
   ```

   For interactive development, you can also install PyBIDS in editable mode:

   ```bash
   pip install -e .
   ```

4. (Optional but recommended) Install the pre-commit hooks so style checks run
   on each commit:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running the tests

Run the test suite through tox. Environments are named `py<XY>-<deps>`, where
`<deps>` is `full` (standard dependencies) or `pre` (pre-release dependencies);
the lowest supported dependency set is available as `py310-min`:

```bash
tox run -e py311-full      # standard run on Python 3.11
tox run -e py310-min       # lowest supported dependencies
tox run -e py314-pre       # pre-release dependencies
```

Tests run under pytest, execute docstring examples (`--doctest-modules`), and
measure coverage over both `src/` and `tests/`. The `min` environment resolves
dependencies to their lowest declared versions, following
[SPEC 0](https://scientific-python.org/specs/spec-0000/); if you raise a
minimum, make sure `py310-min` still passes.

## Test-driven development

If you are fixing a bug, or adding a feature, it is strongly recommended to follow this pattern:

1. Verify that tests pass, e.g., with: `tox -e py314-full`
2. Write a test that fails (again, verify) because it reproduces your bug
   or exercises the feature that does not yet exist.
   This is a code-based specification of the behavior change.
3. Commit the test before doing any further development.
4. Fix the bug or implement the feature, and verify that tests now pass.

Some rules to keep you honest:

1. You can use several commits to implement your changes, but at the end, tests must pass.
2. Changes to the tests must be in their own commits
   and the commit message should explain why the specification has changed.

By developing in this way, you ensure that the feature works as expected,
and that there will be a problem if somebody else breaks the contract you wrote.

## Code style

PyBIDS is formatted and linted with [ruff](https://docs.astral.sh/ruff/) (line
length 99, single quotes; configured in `ruff.toml`). If you installed the
pre-commit hooks, ruff's linter (with autofix) and formatter run automatically;
otherwise run them manually before committing:

```bash
ruff check --fix
ruff format
```

Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/) style.

We track the number of `# noqa` suppressions and fail CI if it grows:

```bash
tox run -e noqa-budget
```

Prefer fixing a lint finding over suppressing it.

**Style-only commits.** Keep formatting-only changes in their own commit,
separate from semantic changes, and include `[ignore-rev]` in the commit
message. You can add the resulting commit hash(es) to
`.git-blame-ignore-revs` so it stays out of `git blame`:

```bash
(
  git log --grep "ignore-rev\]" --pretty=format:"# %ai - %ae - %s%n%H"; echo
) > .git-blame-ignore-revs
```

(We will do this periodically, so this is not a requirement.)

## Commits, branches, and pull requests

- Use short, prefixed commit subjects describing one logical change: `feat:`,
  `fix:`, `doc:`, `rf:` (refactor), `test:`, `style:`, or `chore:`.
- Name branches `<type>/<short-desc>`, e.g. `fix/entity-parsing`.
- Keep each pull request focused on a single logical change; split unrelated
  work into separate PRs.
- Pull requests target the `main` branch.

**Pull request titles become changelog entries.** At release time,
`tools/update_changes.sh` collects merged pull-request titles into
`CHANGELOG.rst`, so write a title that reads well as a one-line summary. You do
not need to edit `CHANGELOG.rst` yourself. The BIDS organization's
`[ENH]`/`[FIX]`/… [PR-title prefixes][bids-contributing] are also welcome if you
prefer them.

## Documentation

Documentation lives in `doc/` and is built with Sphinx:

```bash
tox run -e docs
```

When you change code, check whether any docstrings or documentation describe the
behavior you changed, and update them in the same pull request.
Documentation-only contributions are very welcome.

[bids-contributing]: https://bids-website.readthedocs.io/en/latest/collaboration/bids_github/CONTRIBUTING.html
