# Phase 1 — Data Collection

Phase 1 collects the raw dataset the rest of the project is built on: **C++
questions from Stack Overflow**, gathered via the public Stack Exchange API.

## Contents

| File | Purpose |
| --- | --- |
| `crawl_stackoverflow.py` | Resumable crawler for the Stack Exchange `/search/advanced` API. |
| `stackoverflow_questions.csv` | The collected dataset (~2,500 C++ questions) used by Phase 2 & 3. |
| `P1_brief.pdf`, `P1_brief_datascience.pdf` | The original assignment briefs (reference). |

## The dataset

Each row is one question with its title, HTML body, tags, engagement counts
(views, answers, score), owner info, and timestamps. This single CSV is the
**source of truth** for the whole project — Phase 2 imports it into a database and
Phase 3 trains the recommender on it. The two later phases read it automatically
from `../Phase1/stackoverflow_questions.csv`, so keep the folders together.

## Re-crawling / growing the dataset (optional)

The crawler is **resumable** (it skips question ids already saved and appends to
the CSV) and works **without an API key** (keyless quota ≈ 300 requests/day, i.e.
~30k questions/day). Run it **locally** — CI and cloud sandboxes block outbound
access to `api.stackexchange.com`:

```bash
pip install requests pandas

# One run collects up to ~30k questions (keyless daily quota):
python crawl_stackoverflow.py --pages 300

# Behind a local proxy / VPN:
python crawl_stackoverflow.py --pages 300 --proxy http://127.0.0.1:10808

# With a free app key (raises the quota to 10,000 requests/day):
python crawl_stackoverflow.py --pages 300 --key YOUR_APP_KEY
```

It queries by **tag** (`tagged="c++"`, `sort=creation`, `filter=withbody`) and
**appends** to `stackoverflow_questions.csv` with the exact same column layout, so
nothing downstream needs to change — just re-run Phase 2 / Phase 3 afterwards. The
whole pipeline is size-agnostic, so a larger crawl simply yields a stronger model.
