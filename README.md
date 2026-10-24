# Agentic marketplace on a blockchain

Code of the experiments on LLM agents that bid for jobs in a decentralised
service marketplace. A consumer agent publishes literature-analysis jobs and
sells each in a sealed-bid reverse auction; provider agents bid for the jobs,
execute the ones they win and are paid and rated on chain. The auction scores a
bid by its price and the provider's reputation, with the reputation weight w.
Agent identity and reputation use the ERC-8004 registries deployed on Sepolia.

The experiments compare provider strategies in this market:

- **RQ1:** an LLM that sets the price itself against heuristic pricing rules;
- **RQ2:** an LLM that only estimates a job's cost, with the price set by the
  adaptive heuristic, against non-LLM cost estimators.

## Layout

```
agents/          consumer and provider agents (LangGraph), shared blockchain and IPFS code
  common/        blockchain client, contract ABIs, IPFS, feedback authorisation, dataset replay
  consumer/      publishes jobs, runs their auctions, rates the results
  provider/      bidding strategies, cost estimators, service execution
contracts/       ReverseAuction contract, deployment scripts and tests (Foundry)
experiments/
  conditions/    one runner config per condition, generated; table in conditions/README.md
  runner/        runs one condition at one seed: chain, contracts, agents, metrics
  run_campaign.py  runs (condition, seed) pairs and records them in the ledger
  chain/         the chain snapshot every run starts from, and the script that built it
  dataset/       construction of the recorded service dataset and its paper corpus
analysis/        every figure, table and statistic of the paper
data/            recorded data (not in the repository; see below)
```

## Installation

Python 3.10 and [Foundry](https://book.getfoundry.sh/) (forge and anvil; the
experiments used 1.3.5).

```
git clone --recursive <repository-url>    # or: git submodule update --init
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
cp agents/.env.example agents/.env          # then fill it in; see the comments there
```

`requirements-lock.txt` pins the exact environment the experiments ran in;
`requirements.txt` lists only the direct dependencies, at the same versions. The
contracts are compiled by the runner on first use. OpenZeppelin (v5.4.0) is
a submodule; forge-std is included.

## Data

All data lives under `data/` in the repository, or under the directory named by
the environment variable `MARKETPLACE_DATA_DIR`:

```
data/ledger.jsonl             one line per (condition, seed) run: status, run id, run directory
data/runs/logs/<id>/<run>/    agent logs and status files of each run
data/runs/metrics/<id>/<run>/ experiment_metrics.json of each run
data/dataset/final/           the recorded service dataset the runs replay
data/dataset/artifacts/       per-execution artifacts of the dataset
data/llm_cache/               cache of the cost estimator's LLM calls
data/corpus/                  the source papers of the dataset
```

The recorded data is not part of this repository.

## Reproducing the paper's figures and tables

From the recorded data, without a chain or any LLM call:

```
python -m analysis.make_all
```

Figures are written to `analysis/figures/` (PDF and PNG); every script also
prints the numbers it plots or tabulates, and tables are printed as Markdown and
LaTeX. Each script can be run on its own, e.g.
`python -m analysis.fig_rq2_conditions`. Only runs recorded as `ok` in the ledger
are read.

## Running the experiments

The 48 conditions are listed in
[experiments/conditions/README.md](experiments/conditions/README.md). Each was run
at the 40 seeds 1007–1046, with 30 auctions per run. All configs are generated
from one table over `experiments/conditions/base.yaml`:

```
python -m experiments.generate_conditions --check   # the committed configs match the table
```

One condition at one seed:

```
python experiments/runner/experiment_runner.py experiments/conditions/S91_ladder_band_vs_adaptive.yaml --run-seed 1007
```

The runner starts Anvil from the chain snapshot
(`experiments/chain/sepolia_registries.json`), so a run makes no external RPC
calls. It then deploys the ReverseAuction and a mock USDC, registers the agents
with the ERC-8004 registries, runs the auctions and writes logs and metrics
under `data/runs/`. `--run-index i` isolates concurrent runs (port 8545+i).

A campaign of (condition, seed) pairs, resumable and recorded in
`data/ledger.jsonl`:

```
python experiments/run_campaign.py --seeds 1007 1008 --dry-run
python experiments/run_campaign.py --conditions A_gptoss20b_xd_vs_h --seeds 1007 --parallel 4
```

Pairs already recorded as `ok` are skipped unless `--force`. A run is recorded
as `ok` only if it produced a market: agents started, bids committed, and no
replay failures or late commits.

**Replay of the recorded dataset.** In every run the consumer draws its 30 jobs
from the recorded dataset (`data/dataset/final/`) in an order fixed by the seed.
The winning provider's execution and the consumer's rating are replayed from a
recorded execution of that job, so a run makes no execution or judging LLM call.
The LLM agents' own decisions are live calls through OpenRouter:

- conditions with an LLM price-setter (LLM-Base, -Band, -History, -Reflect) call
  the LLM on every bid, so every run of them is paid;
- LLM-Estimate's calls are cached in `data/llm_cache/`. A re-run replays them only
  until its market diverges from the recorded run (the prompt contains the
  agent's cost history); after that it makes live, paid calls;
- the heuristic, Regression and Oracle conditions make no LLM call, but the agents
  still need a non-empty `OPENROUTER_API_KEY` (any placeholder).

**Runs are not bit-for-bit reproducible.** Gas use depends on how transactions
interleave, and gas feeds the agents' overhead estimate, so bid amounts in a
re-run differ by up to about 1.5%. Winners, ratings and reputations of the
recorded run are reproduced; the final net balance of an agent differs by about
USD 0.001.

## Live mode

The market can also run without the recorded dataset: the consumer generates
each request from papers in a directory, publishes it on IPFS, the winning
provider executes it with its RAG pipeline and the consumer rates the result
with its LLM judge. Every step makes paid LLM calls. Start from any condition
config and change the consumer's behaviour:

```yaml
agents:
  consumer:
    behavior:
      service_source_mode: live_generation
      service_generation_mode: sequential_groups
      service_selection: sequential
      pdf_directory: /absolute/path/to/pdfs     # PDFs, each with its abstract in a sibling .txt
      pdfs_per_service: 2
      num_auctions: 2
```

The live mode needs Pinata (`PINATA_JWT`, or `PINATA_API_KEY` and
`PINATA_API_SECRET`) and a dedicated Pinata gateway (`IPFS_GATEWAY`,
`PINATA_GATEWAY_TOKEN`) in `agents/.env`. The public IPFS gateways do not
reliably serve newly pinned content.

## Rebuilding the inputs

These steps produced the committed inputs; the experiments do not need them.

- **Corpus.** `python -m experiments.dataset.corpus.arxiv_corpus build` downloads
  the 30 arXiv papers pinned in `experiments/dataset/corpus/corpus.yaml` into
  `data/corpus/` and rewrites `manifest.json`, which comes out identical to the
  committed one.
- **Service dataset** (paid LLM calls). `experiments/dataset/collect_dataset.py`
  generates a request for each slot of `requirement_plan.json`, executes each
  request three times and has the consumer's judge rate every execution;
  `finalize_dataset.py` writes the result to `data/dataset/final/`. The two
  invocations are in the module docstring of `collect_dataset.py`, the settings
  in `collection.yaml`.
- **Chain snapshot** (needs `INFURA_API_KEY`).
  `experiments/chain/build_chain_snapshot.py` forks Sepolia once, runs the setup
  so that the ERC-8004 registries are pulled into local state, and dumps the
  chain; usage in its module docstring.
