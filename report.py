from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import Config


def _fmt_pct(x: float) -> str:
    return f"{x:.2%}"


def _fmt_num(x: float) -> str:
    return f"{x:.3f}"


def build_html(rows: list[dict], model: str) -> str:
    cards = []
    for row in rows:
        bt = row["backtest"]
        cards.append(
            f"""
            <article class=\"card\">
              <h3>{row['ticker']}</h3>
              <p class=\"metric\">AUC <span>{_fmt_num(row['auc'])}</span></p>
              <p class=\"metric\">Strategy Return <span>{_fmt_pct(bt['Total Return (strategy)'])}</span></p>
              <p class=\"metric\">Buy&Hold Return <span>{_fmt_pct(bt['Total Return (buy&hold)'])}</span></p>
              <p class=\"metric\">Sharpe <span>{_fmt_num(bt['Sharpe (daily)'])}</span></p>
              <p class=\"metric\">Max Drawdown <span>{_fmt_pct(bt['Max Drawdown'])}</span></p>
            </article>
            """
        )

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
<title>Neural Trading Engine Report</title>
<style>
:root {{
  --bg: #f5f3ef;
  --surface: #fbfaf7;
  --ink: #11212d;
  --muted: #445b66;
  --accent: #0a9396;
  --accent-2: #ee9b00;
  --line: #d8d3c8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: var(--ink);
  background: radial-gradient(circle at 15% 10%, #fffdf8 0%, var(--bg) 60%);
}}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }}
header {{ margin-bottom: 26px; }}
.kicker {{
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
}}
h1 {{ margin: 6px 0 0; font-size: 36px; line-height: 1.1; }}
.subtitle {{ margin-top: 10px; color: var(--muted); }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}}
.card {{
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--surface);
  padding: 16px;
  box-shadow: 0 8px 20px rgba(17,33,45,0.06);
}}
.card h3 {{ margin: 0 0 12px; font-size: 18px; }}
.metric {{
  margin: 8px 0;
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  border-bottom: 1px dashed #e7e2d8;
  padding-bottom: 6px;
}}
.metric span {{ color: var(--ink); font-weight: 700; }}
footer {{ margin-top: 24px; color: var(--muted); font-size: 13px; }}
.tag {{
  display: inline-block;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(120deg, var(--accent), var(--accent-2));
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 12px;
}}
</style>
</head>
<body>
  <div class=\"wrap\">
    <header>
      <p class=\"kicker\">Quant Research Dashboard</p>
      <h1>Neural Trading Engine</h1>
      <p class=\"subtitle\">Model family: <span class=\"tag\">{model.upper()}</span></p>
    </header>
    <section class=\"grid\">
      {''.join(cards)}
    </section>
    <footer>Generated from evaluation artifacts in {Config.artifacts_dir}.</footer>
  </div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "rf", "nn"], required=True)
    args = ap.parse_args()

    fp = Path(Config.artifacts_dir) / f"eval_summary_{args.model}.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Run evaluate.py first.")

    rows = json.loads(fp.read_text())
    html = build_html(rows=rows, model=args.model)

    out = Path(Config.artifacts_dir) / f"dashboard_{args.model}.html"
    out.write_text(html)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
