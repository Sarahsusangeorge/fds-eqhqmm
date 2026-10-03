# Putting this repository on GitHub (one person does this once)

1. On github.com, click **New repository**. Name: `fds-eqhqmm`. Private. Do NOT add a README, .gitignore or licence (this folder already has them).
2. In a terminal, inside this folder:

```bash
git init
git add .
git commit -m "repo: initial structure and shared contracts"
git branch -M main
git remote add origin https://github.com/<your-username>/fds-eqhqmm.git
git push -u origin main
```

3. On GitHub: **Settings → Collaborators → Add people**: add both teammates.
4. **Settings → Branches → Add branch protection rule** for `main`: require a pull request with 1 approval, and require the `tests` check to pass.
5. Edit `.github/CODEOWNERS` with everyone's real GitHub usernames and push that change.

## Each teammate, once

```bash
git clone https://github.com/<owner-username>/fds-eqhqmm.git
cd fds-eqhqmm
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Daily workflow

```bash
git checkout main && git pull
git checkout -b <yourname>/<topic>       # e.g. sarah/engine-filter
# ...work, then:
git add <files>
git commit -m "engine: add filter_step"
git push -u origin <yourname>/<topic>
# open a pull request on GitHub; a teammate reviews and merges
```
