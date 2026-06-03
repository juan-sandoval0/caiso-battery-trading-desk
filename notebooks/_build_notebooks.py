"""Generate the EDA and benchmarking notebooks via nbformat.

Run once to (re)create the .ipynb files:
    python notebooks/_build_notebooks.py

The notebooks are then executed with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent


def _nb(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def build_eda() -> None:
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md(
            "# 01 — Exploratory Data Analysis\n\n"
            "CAISO day-ahead LMPs for the NP15 / SP15 / ZP26 trading hubs.\n"
            "Covers price distributions, day-ahead spreads, and intraday/seasonal "
            "patterns that motivate the battery arbitrage strategy."
        ),
        code(
            "import sys, os\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from src.data.db import MarketDB\n"
            "from src.config.nodes import CAISO_HUB_NODES\n"
            "pd.set_option('display.width', 160)\n"
            "DB_PATH = '../data/market.duckdb'"
        ),
        md("## Load day-ahead LMP for all three hubs"),
        code(
            "with MarketDB(DB_PATH) as db:\n"
            "    latest = db.get_latest_data_date('lmp')\n"
            "    earliest = str(db._conn.execute(\"SELECT MIN(time) FROM lmp WHERE market='DAY_AHEAD_HOURLY'\").fetchone()[0])[:10]\n"
            "    frames = {}\n"
            "    for node in CAISO_HUB_NODES:\n"
            "        frames[node] = db.query_lmp(node, 'DAY_AHEAD_HOURLY', earliest, latest)\n"
            "print('Range:', earliest, '->', latest)\n"
            "{k: len(v) for k, v in frames.items()}"
        ),
        md("## Price distributions by hub"),
        code(
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "for node, df in frames.items():\n"
            "    ax.hist(df['lmp'].clip(-50, 250), bins=80, alpha=0.5, label=node.split('_')[1])\n"
            "ax.set_xlabel('DA LMP ($/MWh)'); ax.set_ylabel('count'); ax.legend(); ax.set_title('DA LMP distribution by hub')\n"
            "plt.tight_layout(); plt.show()\n"
            "pd.DataFrame({node.split('_')[1]: df['lmp'].describe() for node, df in frames.items()})"
        ),
        md(
            "## Daily peak-to-trough spread (NP15)\n\n"
            "The arbitrage opportunity is the daily price spread. The acceptance "
            "criterion expects a typical peak-day NP15 spread of at least ~$20/MWh."
        ),
        code(
            "np15 = frames['TH_NP15_GEN-APND'].copy()\n"
            "np15['time'] = pd.to_datetime(np15['time'])\n"
            "np15['date'] = np15['time'].dt.date\n"
            "daily = np15.groupby('date')['lmp'].agg(['min', 'max'])\n"
            "daily['spread'] = daily['max'] - daily['min']\n"
            "print('Median daily spread: $%.1f/MWh' % daily['spread'].median())\n"
            "print('Share of days with spread >= $20/MWh: %.0f%%' % (100 * (daily['spread'] >= 20).mean()))\n"
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "ax.hist(daily['spread'].clip(0, 200), bins=60)\n"
            "ax.axvline(20, color='red', ls='--', label='$20/MWh')\n"
            "ax.set_xlabel('Daily DA spread ($/MWh)'); ax.set_ylabel('days'); ax.legend(); ax.set_title('NP15 daily DA price spread')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## Average intraday price shape (NP15)"),
        code(
            "np15['hour'] = np15['time'].dt.hour\n"
            "hourly = np15.groupby('hour')['lmp'].mean()\n"
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "ax.plot(hourly.index, hourly.values, marker='o')\n"
            "ax.set_xlabel('hour of day'); ax.set_ylabel('mean DA LMP ($/MWh)'); ax.set_title('NP15 average intraday price shape')\n"
            "plt.tight_layout(); plt.show()\n"
            "hourly"
        ),
        md(
            "## Takeaways\n\n"
            "- All three hubs show a heavy right tail (scarcity pricing) and occasional negative prices (solar over-supply).\n"
            "- The NP15 daily DA spread is regularly well above the $20/MWh arbitrage threshold, especially in summer.\n"
            "- The classic duck-curve shape (cheap midday solar, expensive evening ramp) is the structural edge the battery exploits."
        ),
    ]
    nbf.write(_nb(cells), str(_HERE / "01_eda.ipynb"))
    print("wrote 01_eda.ipynb")


def build_benchmarking() -> None:
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md(
            "# 02 — Strategy Benchmarking\n\n"
            "Evaluates the multi-agent forecaster + dispatch optimizer against the "
            "naive valley-fill baseline and the perfect-hindsight upper bound, using "
            "the backtest results produced by `python main.py backtest`."
        ),
        code(
            "import sys, os\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "CSV = '../results/backtest_2024H2.csv'\n"
            "bt = pd.read_csv(CSV, parse_dates=['date'])\n"
            "bt.head()"
        ),
        md("## Cumulative P&L vs. baselines"),
        code(
            "fig, ax = plt.subplots(figsize=(11, 5))\n"
            "ax.plot(bt['date'], bt['revenue_usd'].cumsum(), label='Our system')\n"
            "ax.plot(bt['date'], bt['naive_revenue_usd'].cumsum(), label='Naive valley-fill')\n"
            "ax.plot(bt['date'], bt['perfect_revenue_usd'].cumsum(), label='Perfect hindsight')\n"
            "ax.set_ylabel('cumulative $'); ax.legend(); ax.set_title('Cumulative P&L')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## Headline performance metrics"),
        code(
            "rev = bt['revenue_usd'].to_numpy(float)\n"
            "naive = bt['naive_revenue_usd'].to_numpy(float)\n"
            "perfect = bt['perfect_revenue_usd'].to_numpy(float)\n"
            "cum = np.cumsum(rev)\n"
            "sharpe = rev.mean() / rev.std() * np.sqrt(252) if rev.std() > 0 else float('nan')\n"
            "max_dd = float(np.max(np.maximum.accumulate(cum) - cum))\n"
            "metrics = {\n"
            "    'days': len(bt),\n"
            "    'total_revenue_usd': rev.sum(),\n"
            "    'naive_revenue_usd': naive.sum(),\n"
            "    'perfect_revenue_usd': perfect.sum(),\n"
            "    'vs_naive_pct': (rev.sum() - naive.sum()) / abs(naive.sum()) * 100 if naive.sum() else float('nan'),\n"
            "    'pct_of_perfect': rev.sum() / perfect.sum() * 100 if perfect.sum() else float('nan'),\n"
            "    'sharpe_annualized': sharpe,\n"
            "    'max_drawdown_usd': max_dd,\n"
            "    'win_rate_pct': (rev > 0).mean() * 100,\n"
            "    'mean_forecast_rmse': bt['forecast_rmse'].mean(),\n"
            "}\n"
            "pd.Series(metrics).round(2)"
        ),
        md("## Win rate by hour-of-day\n\nFraction of days the strategy earned positive revenue, and how cycles distribute."),
        code(
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
            "ax[0].hist(rev, bins=40); ax[0].axvline(0, color='red', ls='--')\n"
            "ax[0].set_title('Daily revenue distribution'); ax[0].set_xlabel('$ / day')\n"
            "ax[1].plot(bt['date'], bt['cycles']); ax[1].set_title('Daily cycles'); ax[1].set_ylabel('equivalent full cycles')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "## Conclusion\n\n"
            "The multi-agent system is benchmarked against the naive valley-fill strategy "
            "(target: beat it by >=15%) and reported as a fraction of the perfect-hindsight "
            "upper bound. See the headline metrics table above for the realized figures on "
            "the 2024 H2 out-of-sample window."
        ),
    ]
    nbf.write(_nb(cells), str(_HERE / "02_benchmarking.ipynb"))
    print("wrote 02_benchmarking.ipynb")


if __name__ == "__main__":
    build_eda()
    build_benchmarking()
