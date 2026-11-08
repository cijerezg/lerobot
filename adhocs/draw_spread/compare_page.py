"""One page with the draw-spread runs side by side: the comparison figures on top, then one column per run with
its headline table and figures (calibration/scale scatters, spread vs error absolute and relative, val by verb and
family, style projection), with links to each run's report and hover page.

    uv run python lerobot/adhocs/draw_spread/compare_page.py <compare_dir> <label>=<run_dir> [<label>=<run_dir> ...]
(same arguments as compare_runs.py, run that first)
"""
import base64, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from report import CSS, fig_scatters, fig_style, table_headline  # noqa: E402

OUT = sys.argv[1]
runs = [arg.split("=", 1) for arg in sys.argv[2:]]


def img_file(path: str) -> str:
    if not os.path.exists(path):
        return f"<p class=note>missing {os.path.basename(path)}</p>"
    return f"<img src='data:image/png;base64,{base64.b64encode(open(path, 'rb').read()).decode()}'>"


def rel(path: str) -> str:
    return os.path.relpath(path, OUT)


summary = pd.read_csv(f"{OUT}/summary.csv")
cols = ["run", "set", "anchors", "rms spread", "rms error of mean", "error/spread (median of ratio)", "error/spread q25", "error/spread q75", "R (mse/hold)", "spread std / data std", "style proj offset", "corr(log spread, log error)"]
summary_html = summary[cols].round(3).to_html(index=False, border=0)
verb_html = pd.read_csv(f"{OUT}/ratio_by_verb.csv").round(2).to_html(index=False, border=0)

columns = []
for label, d in runs:
    df = pd.read_csv(f"{d}/anchors.csv")
    df["episode"] = df.apply(lambda r: f"{os.path.basename(r.root)} ep{r.episode}", axis=1)
    clean = df[df.setting == "clean"]
    parts = [f"<h2>{label}</h2>",
             f"<p class=note><a href='{rel(d + '/report.html')}'>report</a> · <a href='{rel(d + '/hover.html')}'>hover page</a> · {os.path.basename(d)}</p>",
             table_headline(clean),
             "<h3>calibration and scale</h3>", f"<img src='{fig_scatters(clean)}'>",
             "<h3>spread vs error</h3>", img_file(f"{d}/spread_vs_error.png"),
             "<h3>spread vs error, relative to motion</h3>", img_file(f"{d}/spread_vs_error_relative.png"),
             "<h3>val by verb</h3>", img_file(f"{d}/val_spread_vs_error_by_verb.png"),
             "<h3>val by object family</h3>", img_file(f"{d}/val_spread_vs_error_by_family.png"),
             "<h3>style projection</h3>", f"<img src='{fig_style(clean)}'>"]
    columns.append("<div class=col>" + "\n".join(parts) + "</div>")

css = CSS + """
body{max-width:none;margin:16px}
.grid{display:grid;grid-template-columns:repeat(COLS,minmax(0,1fr));gap:18px;align-items:start}
.col{min-width:0;border-top:3px solid #e1e0d9;padding-top:4px}.col h2{margin-top:4px}
.col img{width:100%}.col table{font-size:11px}.col th,.col td{padding:2px 5px}
h3{font-size:13px;margin:14px 0 4px;color:#52514e}
""".replace("COLS", str(len(runs)))
html = [f"<!doctype html><html><head><meta charset=utf-8><title>Draw spread, checkpoint comparison</title><style>{css}</style></head><body>",
        "<h1>Draw spread, checkpoint comparison</h1>",
        f"<p class=note>runs: {', '.join(f'{l} = {os.path.basename(d)}' for l, d in runs)}. Same 9 val episodes and 8 training episodes; clean inputs; 64 draws per anchor.</p>",
        "<h2>Across runs</h2>", summary_html, "<p class=note>medians over anchors; rms in normalized action units.</p>",
        img_file(f"{OUT}/compare_spread_vs_error.png"), img_file(f"{OUT}/compare_ratio.png"),
        "<h3>error / spread by verb</h3>", verb_html,
        "<h2>Per run, side by side</h2>", "<div class=grid>" + "\n".join(columns) + "</div>",
        "</body></html>"]
open(f"{OUT}/compare.html", "w").write("\n".join(html))
print(f"wrote {OUT}/compare.html ({os.path.getsize(f'{OUT}/compare.html') / 1e6:.1f} MB)")
