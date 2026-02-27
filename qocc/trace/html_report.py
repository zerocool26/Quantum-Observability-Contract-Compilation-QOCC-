"""Interactive HTML trace report export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qocc.core.artifacts import ArtifactStore


def export_html_report(
    bundle_path: str,
    output_path: str,
    compare_bundle_path: str | None = None,
) -> Path:
    """Export a self-contained HTML report for a trace bundle.

    Parameters:
        bundle_path: Path to bundle zip or extracted directory.
        output_path: Destination HTML file path.
        compare_bundle_path: Optional second bundle for diff view.

    Returns:
        Path to generated HTML report.
    """
    bundle = ArtifactStore.load_bundle(bundle_path)
    root = Path(bundle.get("_root", ""))

    search_rankings = _read_json_if_exists(root / "search_rankings.json")
    compare_report = None
    if compare_bundle_path:
        from qocc.api import compare_bundles

        compare_report = compare_bundles(bundle_path, compare_bundle_path)

    payload = {
        "manifest": bundle.get("manifest", {}),
        "spans": bundle.get("trace", []),
        "metrics": bundle.get("metrics", {}),
        "contract_results": bundle.get("contract_results", []),
        "search_rankings": search_rankings if isinstance(search_rankings, list) else [],
        "compare": compare_report,
    }

    html = _build_html(payload)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), default=str)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>QOCC Trace Report</title>
  <style>
    :root {{ --bg:#0b1020; --panel:#131a2b; --text:#e8eefc; --muted:#9eb0d0; --line:#25314f; --ok:#22c55e; --bad:#ef4444; --blue:#3b82f6; --green:#16a34a; --orange:#f59e0b; --gray:#64748b; }}
    body {{ margin:0; padding:16px; font-family:Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--text); }}
    h1,h2 {{ margin:0 0 10px; }}
    .section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin:12px 0; }}
    .row {{ display:flex; gap:10px; align-items:center; }}
    .muted {{ color:var(--muted); }}
    .badge {{ font-size:12px; border-radius:999px; padding:2px 8px; border:1px solid var(--line); }}
    .ok {{ color:var(--ok); border-color:var(--ok); }}
    .bad {{ color:var(--bad); border-color:var(--bad); }}
    .flame-wrap {{ border:1px solid var(--line); border-radius:6px; padding:8px; overflow-x:auto; }}
    .flame-row {{ display:grid; grid-template-columns: 260px 1fr; align-items:center; margin:4px 0; gap:8px; }}
    .track {{ position:relative; height:20px; background:#0f1629; border:1px solid #1e2a47; border-radius:4px; }}
    .bar {{ position:absolute; top:1px; bottom:1px; border-radius:3px; opacity:0.9; }}
    .bar-label {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .bars .item {{ margin:8px 0; }}
    .meter {{ height:12px; background:#0f1629; border:1px solid #1e2a47; border-radius:999px; overflow:hidden; }}
    .meter > div {{ height:100%; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:6px; text-align:left; }}
    .ci {{ height:10px; background:#0f1629; border:1px solid #1e2a47; border-radius:999px; position:relative; }}
    .ci > div {{ position:absolute; top:1px; bottom:1px; border-radius:999px; background:var(--green); }}
    .legend span {{ margin-right:10px; }}
  </style>
</head>
<body>
  <h1>QOCC Trace Report</h1>
  <div class=\"muted\" id=\"summary\"></div>

  <div class=\"section\">
    <h2>Flame Chart</h2>
    <div class=\"legend muted\">
      <span>Adapters: blue</span><span>Contracts: green</span><span>Search: orange</span><span>Cache: gray</span>
    </div>
    <div class=\"flame-wrap\" id=\"flame\"></div>
  </div>

  <div class=\"section\">
    <h2>Metric Dashboard</h2>
    <div class=\"bars\" id=\"metrics\"></div>
  </div>

  <div class=\"section\">
    <h2>Contract Results</h2>
    <table id=\"contracts\"><thead><tr><th>Name</th><th>Status</th><th>CI</th></tr></thead><tbody></tbody></table>
  </div>

  <div class=\"section\" id=\"diffSection\" style=\"display:none\">
    <h2>Diff View</h2>
    <table id=\"diffTable\"><thead><tr><th>Metric</th><th>A</th><th>B</th><th>Change</th></tr></thead><tbody></tbody></table>
  </div>

  <div class=\"section\">
    <h2>Circuit Diff</h2>
    <table id=\"histTable\"><thead><tr><th>Gate</th><th>Input</th><th>Selected</th></tr></thead><tbody></tbody></table>
  </div>

  <script>
    const DATA = {payload};

    const moduleColor = (name) => {{
      const n = (name || '').toLowerCase();
      if (n.includes('contract')) return 'var(--green)';
      if (n.includes('search') || n.includes('score') || n.includes('candidate')) return 'var(--orange)';
      if (n.includes('cache')) return 'var(--gray)';
      return 'var(--blue)';
    }};

    const summary = () => {{
      const m = DATA.manifest || {{}};
      const txt = `Run: ${{m.run_id || 'unknown'}} · Adapter: ${{m.adapter || 'unknown'}} · Created: ${{m.created_at || 'unknown'}}`;
      document.getElementById('summary').textContent = txt;
    }};

    const flame = () => {{
      const spans = Array.isArray(DATA.spans) ? DATA.spans : [];
      const host = document.getElementById('flame');
      if (!spans.length) {{ host.textContent = '(no spans)'; return; }}

      const parsed = spans.map((s) => {{
        const start = Date.parse(s.start_time || 0) / 1000 || Number(s.start_time || 0);
        const endRaw = s.end_time != null ? s.end_time : s.start_time;
        const end = Date.parse(endRaw || 0) / 1000 || Number(endRaw || start);
        return {{ name: s.name || '?', attrs: s.attributes || {{}}, start, end: Math.max(end, start), parent: s.parent_span_id || s.parent_id || null }};
      }});

      const min = Math.min(...parsed.map((s) => s.start));
      const max = Math.max(...parsed.map((s) => s.end));
      const range = Math.max(max - min, 1e-9);

      parsed.sort((a, b) => a.start - b.start);
      for (const s of parsed) {{
        const row = document.createElement('div');
        row.className = 'flame-row';

        const label = document.createElement('div');
        label.className = 'bar-label';
        label.textContent = s.name;

        const track = document.createElement('div');
        track.className = 'track';

        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.left = `${{((s.start - min) / range) * 100}}%`;
        bar.style.width = `${{Math.max(((s.end - s.start) / range) * 100, 0.5)}}%`;
        bar.style.background = moduleColor(s.name);
        bar.title = JSON.stringify(s.attrs, null, 2);

        track.appendChild(bar);
        row.appendChild(label);
        row.appendChild(track);
        host.appendChild(row);
      }}
    }};

    const metricDashboard = () => {{
      const host = document.getElementById('metrics');
      const rankings = Array.isArray(DATA.search_rankings) ? DATA.search_rankings : [];
      let rows = [];

      if (rankings.length) {{
        rows = rankings.map((r, idx) => ({{
          name: r.candidate_id || `candidate_${{idx+1}}`,
          depth: Number((r.metrics || {{}}).depth || 0),
          gates2q: Number((r.metrics || {{}}).gates_2q || 0),
          error: Number((r.metrics || {{}}).proxy_error_score || 0),
          duration: Number((r.metrics || {{}}).duration_estimate || 0),
        }}));
      }} else {{
        const m = DATA.metrics || {{}};
        const c = m.compiled || {{}};
        rows = [{{
          name: 'selected',
          depth: Number(c.depth || 0),
          gates2q: Number(c.gates_2q || 0),
          error: Number(c.proxy_error_score || 0),
          duration: Number(c.duration_estimate || 0),
        }}];
      }}

      const keys = ['depth', 'gates2q', 'error', 'duration'];
      for (const key of keys) {{
        const max = Math.max(1, ...rows.map((r) => Number(r[key] || 0)));
        const section = document.createElement('div');
        section.className = 'item';
        const title = document.createElement('div');
        title.className = 'muted';
        title.textContent = key;
        section.appendChild(title);

        for (const r of rows) {{
          const v = Number(r[key] || 0);
          const line = document.createElement('div');
          line.className = 'row';
          line.innerHTML = `<div style=\"width:180px\">${{r.name}}</div><div class=\"meter\" style=\"flex:1\"><div style=\"width:${{(v/max)*100}}%;background:var(--blue)\"></div></div><div style=\"width:70px;text-align:right\">${{v.toFixed(3)}}</div>`;
          section.appendChild(line);
        }}
        host.appendChild(section);
      }}
    }};

    const contracts = () => {{
      const tbody = document.querySelector('#contracts tbody');
      const rows = Array.isArray(DATA.contract_results) ? DATA.contract_results : [];
      if (!rows.length) {{
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="3" class="muted">(no contract results)</td>';
        tbody.appendChild(tr);
        return;
      }}

      for (const r of rows) {{
        const passed = !!r.passed;
        const details = r.details || {{}};
        const ci = details.confidence_interval || details.ci || null;
        let ciHtml = '<span class="muted">n/a</span>';
        if (Array.isArray(ci) && ci.length === 2) {{
          const lo = Number(ci[0]);
          const hi = Number(ci[1]);
          const left = Math.max(0, Math.min(100, lo * 100));
          const width = Math.max(1, Math.min(100 - left, (hi - lo) * 100));
          ciHtml = `<div class=\"ci\"><div style=\"left:${{left}}%;width:${{width}}%\"></div></div>`;
        }}

        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{r.name || 'contract'}}</td><td><span class=\"badge ${{passed ? 'ok' : 'bad'}}\">${{passed ? 'PASS' : 'FAIL'}}</span></td><td>${{ciHtml}}</td>`;
        tbody.appendChild(tr);
      }}
    }};

    const diff = () => {{
      const report = DATA.compare;
      if (!report || !report.diffs) return;
      const compiled = (((report.diffs || {{}}).metrics || {{}}).compiled || {{}});
      const tbody = document.querySelector('#diffTable tbody');
      const keys = Object.keys(compiled);
      if (!keys.length) return;

      document.getElementById('diffSection').style.display = 'block';
      for (const key of keys) {{
        const d = compiled[key] || {{}};
        const pct = typeof d.pct_change === 'number' ? `${{d.pct_change.toFixed(2)}}%` : '-';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{key}}</td><td>${{String(d.a)}}</td><td>${{String(d.b)}}</td><td>${{pct}}</td>`;
        tbody.appendChild(tr);
      }}
    }};

    const circuitDiff = () => {{
      const m = DATA.metrics || {{}};
      const a = ((m.input || {{}}).gate_histogram || {{}});
      const b = ((m.compiled || {{}}).gate_histogram || {{}});
      const keys = Array.from(new Set([...Object.keys(a), ...Object.keys(b)])).sort();
      const tbody = document.querySelector('#histTable tbody');

      if (!keys.length) {{
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="3" class="muted">(no gate histograms)</td>';
        tbody.appendChild(tr);
        return;
      }}

      for (const k of keys) {{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{k}}</td><td>${{a[k] || 0}}</td><td>${{b[k] || 0}}</td>`;
        tbody.appendChild(tr);
      }}
    }};

    summary();
    flame();
    metricDashboard();
    contracts();
    diff();
    circuitDiff();
  </script>
</body>
</html>"""
