#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


SPAN_RE = re.compile(r"^(?P<token>.*)\[(?P<start>\d+):(?P<end>\d+)\]$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_span_list(cell: str) -> list[dict[str, int | str]]:
    items = []
    for piece in (cell or "").split(" | "):
        piece = piece.strip()
        if not piece:
            continue
        m = SPAN_RE.match(piece)
        if not m:
            continue
        items.append(
            {
                "token": m.group("token"),
                "start": int(m.group("start")),
                "end": int(m.group("end")),
            }
        )
    return items


def token_spans(segmented: str) -> list[dict[str, int | str]]:
    pos = 0
    spans = []
    for token in (segmented or "").split():
        start = pos
        end = pos + len(token)
        spans.append({"token": token, "start": start, "end": end})
        pos = end
    return spans


def classify_word_order_flags(
    target_text: str,
    gold_only: list[dict[str, int | str]],
    pred_only: list[dict[str, int | str]],
) -> dict[str, object]:
    target_chars = (target_text or "").replace(" ", "")
    gold_hits = [item["token"] for item in gold_only if str(item["token"]) in target_chars]
    pred_hits = [item["token"] for item in pred_only if str(item["token"]) in target_chars]
    return {
        "has_word_order_source": bool(gold_hits),
        "has_word_order_pred": bool(pred_hits),
        "has_word_order_any": bool(gold_hits or pred_hits),
        "has_word_order_strict": bool(gold_hits and pred_hits),
        "word_order_source_tokens": gold_hits,
        "word_order_pred_tokens": pred_hits,
    }


def build_records(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    records = []
    for row in rows:
        gold_only = parse_span_list(row.get("Gold_Only_Tokens", ""))
        pred_only = parse_span_list(row.get("Pred_Only_Tokens", ""))
        gold_tokens = token_spans(row.get("Gold_Source", ""))
        pred_tokens = token_spans(row.get("Projected_Source", ""))
        word_order = classify_word_order_flags(row.get("Target_Text", ""), gold_only, pred_only)

        records.append(
            {
                "index": row.get("Index", ""),
                "id": row.get("ID", ""),
                "token_precision": float(row.get("Token_Precision", "0") or 0),
                "token_recall": float(row.get("Token_Recall", "0") or 0),
                "token_f1": float(row.get("Token_F1", "0") or 0),
                "gold_source": row.get("Gold_Source", ""),
                "target_text": row.get("Target_Text", ""),
                "projected_source": row.get("Projected_Source", ""),
                "gold_only": gold_only,
                "pred_only": pred_only,
                "gold_tokens": gold_tokens,
                "target_tokens": token_spans(row.get("Target_Text", "")),
                "pred_tokens": pred_tokens,
                "added_boundaries": row.get("Added_Boundaries", ""),
                "deleted_boundaries": row.get("Deleted_Boundaries", ""),
                "projected_boundary_count": row.get("Projected_Boundary_Count", ""),
                "merged_spans": row.get("Merged_Spans", ""),
                "merge_skips": row.get("Merge_Skips", ""),
                "skipped_boundaries": row.get("Skipped_Boundaries", ""),
                **word_order,
            }
        )
    return records


def build_html(records: list[dict[str, object]], title: str) -> str:
    data_json = json.dumps(records, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffdf8;
      --ink: #1e2430;
      --muted: #68707f;
      --line: #d9ceb9;
      --gold-bad: #ffd7d2;
      --pred-bad: #d7e7ff;
      --good: #4c7b62;
      --accent: #b6453c;
      --shadow: 0 18px 40px rgba(58, 42, 18, 0.08);
      --font: "Avenir Next", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      --mono: "SF Mono", "Menlo", monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(199, 124, 69, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(50, 93, 168, 0.12), transparent 24%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      align-items: end;
      margin-bottom: 22px;
    }}
    .hero-card, .panel {{
      background: rgba(255,255,255,0.82);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(217, 206, 185, 0.9);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero-card {{
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 38px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .lead {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 20px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .stat-value {{
      font-size: 30px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1.05fr 0.55fr 0.55fr 0.55fr 0.7fr;
      gap: 14px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .field {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .field label {{
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .field input, .field select {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
      font: inherit;
      color: var(--ink);
    }}
    .results {{
      display: grid;
      gap: 16px;
    }}
    .card {{
      padding: 18px;
      border: 1px solid rgba(217, 206, 185, 0.85);
      border-radius: 24px;
      background: rgba(255,255,255,0.85);
      box-shadow: var(--shadow);
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .title-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .id {{
      font-weight: 700;
      font-size: 18px;
    }}
    .score {{
      text-align: right;
      min-width: 120px;
    }}
    .score strong {{
      display: block;
      font-size: 34px;
      letter-spacing: -0.04em;
      color: var(--accent);
    }}
    .score span {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .token-panes {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .pane {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .pane h3 {{
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .tokens {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      line-height: 1.6;
    }}
    .token {{
      border-radius: 999px;
      border: 1px solid rgba(127, 136, 149, 0.22);
      background: #fff;
      padding: 6px 10px;
      font-size: 15px;
    }}
    .token.bad-gold {{
      background: var(--gold-bad);
      border-color: #e79e96;
    }}
    .token.bad-pred {{
      background: var(--pred-bad);
      border-color: #97b6e8;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .meta {{
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 12px;
      min-height: 82px;
    }}
    .meta .k {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .meta .v {{
      font-family: var(--mono);
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .empty {{
      padding: 32px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 24px;
      background: rgba(255,255,255,0.55);
    }}
    .footer-note {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1080px) {{
      .hero,
      .controls,
      .token-panes,
      .meta-grid {{
        grid-template-columns: 1fr;
      }}
      .card-top {{
        flex-direction: column;
      }}
      .score {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-card">
        <h1>Token Mismatch Viewer</h1>
        <p class="lead">Compare <code>Gold_Source</code> and <code>Projected_Source</code> token by token. Red tokens appear only on the gold side, while blue tokens appear only on the predicted side or reflect a token boundary mismatch.</p>
      </div>
      <div class="hero-card">
        <div class="stat-grid">
          <div class="stat">
            <span class="stat-label">Mismatch Examples</span>
            <span class="stat-value" id="statRows">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">Average Token F1</span>
            <span class="stat-value" id="statF1">0.0000</span>
          </div>
          <div class="stat">
            <span class="stat-label">Gold-Only Tokens</span>
            <span class="stat-value" id="statGoldOnly">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">Pred-Only Tokens</span>
            <span class="stat-value" id="statPredOnly">0</span>
          </div>
        </div>
      </div>
    </section>

    <section class="panel controls">
      <div class="field">
        <label for="search">Search</label>
        <input id="search" type="text" placeholder="Search by ID, token, or sentence text">
      </div>
      <div class="field">
        <label for="maxF1">Max Token F1</label>
        <input id="maxF1" type="number" min="0" max="1" step="0.01" value="1.00">
      </div>
      <div class="field">
        <label for="sortBy">Sort By</label>
        <select id="sortBy">
          <option value="f1Asc">Token F1 Low to High</option>
          <option value="f1Desc">Token F1 High to Low</option>
          <option value="indexAsc">Index Low to High</option>
          <option value="indexDesc">Index High to Low</option>
        </select>
      </div>
      <div class="field">
        <label for="wordOrderFilter">Word Order</label>
        <select id="wordOrderFilter">
          <option value="all">All mismatches</option>
          <option value="any">Word-order candidates</option>
          <option value="strict">Strict word-order candidates</option>
          <option value="source">Source-side candidates</option>
          <option value="pred">Projected-side candidates</option>
        </select>
      </div>
      <div class="field">
        <label for="limit">Show</label>
        <select id="limit">
          <option value="25">25 rows</option>
          <option value="50" selected>50 rows</option>
          <option value="100">100 rows</option>
          <option value="9999">All rows</option>
        </select>
      </div>
    </section>

    <div id="results" class="results"></div>
    <div class="footer-note">Tip: lower <code>Max Token F1</code> to around 0.85 first if you want to inspect the worst examples.</div>
  </div>

  <script>
    const DATA = {data_json};

    const els = {{
      search: document.getElementById("search"),
      maxF1: document.getElementById("maxF1"),
      sortBy: document.getElementById("sortBy"),
      wordOrderFilter: document.getElementById("wordOrderFilter"),
      limit: document.getElementById("limit"),
      results: document.getElementById("results"),
      statRows: document.getElementById("statRows"),
      statF1: document.getElementById("statF1"),
      statGoldOnly: document.getElementById("statGoldOnly"),
      statPredOnly: document.getElementById("statPredOnly"),
    }};

    function esc(text) {{
      return String(text ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;");
    }}

    function tokenKey(item) {{
      return `${{item.token}}|${{item.start}}|${{item.end}}`;
    }}

    function renderTokens(tokens, badTokens, cls) {{
      const badSet = new Set((badTokens || []).map(tokenKey));
      return tokens.map((tok) => {{
        const bad = badSet.has(tokenKey(tok)) ? ` ${{cls}}` : "";
        const title = `${{tok.token}} [${{tok.start}}:${{tok.end}}]`;
        return `<span class="token${{bad}}" title="${{esc(title)}}">${{esc(tok.token)}}</span>`;
      }}).join("");
    }}

    function metaBox(label, value) {{
      return `<div class="meta"><div class="k">${{esc(label)}}</div><div class="v">${{esc(value || "-")}}</div></div>`;
    }}

    function matchesSearch(row, query) {{
      if (!query) return true;
      const hay = [
        row.index, row.id, row.gold_source, row.target_text, row.projected_source,
        ...(row.gold_only || []).map(x => x.token),
        ...(row.pred_only || []).map(x => x.token)
      ].join("\\n").toLowerCase();
      return hay.includes(query);
    }}

    function matchesWordOrder(row, mode) {{
      if (mode === "all") return true;
      if (mode === "any") return row.has_word_order_any;
      if (mode === "strict") return row.has_word_order_strict;
      if (mode === "source") return row.has_word_order_source;
      if (mode === "pred") return row.has_word_order_pred;
      return true;
    }}

    function sortRows(rows, mode) {{
      const copy = [...rows];
      const indexNum = (x) => Number(x.index || 0);
      if (mode === "f1Desc") copy.sort((a, b) => b.token_f1 - a.token_f1 || indexNum(a) - indexNum(b));
      if (mode === "indexAsc") copy.sort((a, b) => indexNum(a) - indexNum(b));
      if (mode === "indexDesc") copy.sort((a, b) => indexNum(b) - indexNum(a));
      if (mode === "f1Asc") copy.sort((a, b) => a.token_f1 - b.token_f1 || indexNum(a) - indexNum(b));
      return copy;
    }}

    function render() {{
      const query = els.search.value.trim().toLowerCase();
      const maxF1 = Number(els.maxF1.value || "1");
      const limit = Number(els.limit.value || "50");
      const wordOrderMode = els.wordOrderFilter.value;
      const sorted = sortRows(
        DATA.filter((row) =>
          row.token_f1 <= maxF1 &&
          matchesSearch(row, query) &&
          matchesWordOrder(row, wordOrderMode)
        ),
        els.sortBy.value
      );
      const visible = sorted.slice(0, limit);

      els.statRows.textContent = String(sorted.length);
      const avgF1 = sorted.length ? (sorted.reduce((s, r) => s + r.token_f1, 0) / sorted.length) : 0;
      const goldOnly = sorted.reduce((s, r) => s + r.gold_only.length, 0);
      const predOnly = sorted.reduce((s, r) => s + r.pred_only.length, 0);
      els.statF1.textContent = avgF1.toFixed(4);
      els.statGoldOnly.textContent = String(goldOnly);
      els.statPredOnly.textContent = String(predOnly);

      if (!visible.length) {{
        els.results.innerHTML = '<div class="empty">No examples match the current filters.</div>';
        return;
      }}

      els.results.innerHTML = visible.map((row) => `
        <article class="card">
          <div class="card-top">
            <div>
              <div class="title-line">
                <span class="id">Index ${{esc(row.index)}} / ID ${{esc(row.id)}}</span>
                <span class="badge">P ${{row.token_precision.toFixed(4)}}</span>
                <span class="badge">R ${{row.token_recall.toFixed(4)}}</span>
              </div>
              <div class="title-line">
                <span class="badge">Gold-only tokens: ${{row.gold_only.length}}</span>
                <span class="badge">Pred-only tokens: ${{row.pred_only.length}}</span>
                <span class="badge">Word-order: ${{row.has_word_order_any ? "yes" : "no"}}</span>
              </div>
            </div>
            <div class="score">
              <span>Token F1</span>
              <strong>${{row.token_f1.toFixed(4)}}</strong>
            </div>
          </div>

          <div class="token-panes">
            <section class="pane">
              <h3>Target Text</h3>
              <div class="tokens">${{renderTokens(row.target_tokens, [], "")}}</div>
            </section>
            <section class="pane">
              <h3>Gold Source</h3>
              <div class="tokens">${{renderTokens(row.gold_tokens, row.gold_only, "bad-gold")}}</div>
            </section>
            <section class="pane">
              <h3>Projected Source</h3>
              <div class="tokens">${{renderTokens(row.pred_tokens, row.pred_only, "bad-pred")}}</div>
            </section>
          </div>

          <div class="meta-grid">
            ${{metaBox("Gold Only Tokens", (row.gold_only || []).map(x => `${{x.token}} [${{x.start}}:${{x.end}}]`).join("\\n"))}}
            ${{metaBox("Pred Only Tokens", (row.pred_only || []).map(x => `${{x.token}} [${{x.start}}:${{x.end}}]`).join("\\n"))}}
            ${{metaBox("Word-Order Source Tokens", (row.word_order_source_tokens || []).join("\\n"))}}
            ${{metaBox("Word-Order Pred Tokens", (row.word_order_pred_tokens || []).join("\\n"))}}
            ${{metaBox("Added Boundaries", row.added_boundaries)}}
            ${{metaBox("Deleted Boundaries", row.deleted_boundaries)}}
            ${{metaBox("Projected Boundary Count", row.projected_boundary_count)}}
            ${{metaBox("Merged Spans", row.merged_spans)}}
            ${{metaBox("Merge Skips", row.merge_skips)}}
            ${{metaBox("Skipped Boundaries", row.skipped_boundaries)}}
          </div>
        </article>
      `).join("");
    }}

    ["input", "change"].forEach((eventName) => {{
      els.search.addEventListener(eventName, render);
      els.maxF1.addEventListener(eventName, render);
      els.sortBy.addEventListener(eventName, render);
      els.wordOrderFilter.addEventListener(eventName, render);
      els.limit.addEventListener(eventName, render);
    }});

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    csv_path = (
        script_dir / "../data/0415/step2_weight_optimization_spanlen_cv/cv_best_step2_token_mismatch_examples.csv"
    ).resolve()
    out_path = csv_path.with_name("cv_best_step2_token_mismatch_viewer.html")

    records = build_records(read_csv(csv_path))
    html = build_html(records, "CV Best Step2 Token Mismatch Viewer")
    out_path.write_text(html, encoding="utf-8")

    print(f"Saved viewer: {out_path}")
    print(f"Loaded mismatch examples: {len(records)}")


if __name__ == "__main__":
    main()
