"""
谋杀湾·数据与评估可视化报告生成器
====================================

将 data_report.json 和 eval_report.json（可选）渲染为
一份自包含的 HTML 可视化报告，无需任何服务器或额外依赖。

用法：
  python scripts/visualize_report.py
  python scripts/visualize_report.py --output reports/report.html
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ══════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════

def load_json(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════
# HTML 模板
# ══════════════════════════════════════════════

def build_html(data_report: Dict, eval_report: Optional[List]) -> str:
    ps = data_report["pipeline_summary"]
    cat_dist = data_report["category_distribution"]
    verb_dist = data_report["verb_distribution_top30"]
    tool_dist = data_report["tool_function_distribution"]
    len_stats = data_report["output_length_stats"]
    balance = data_report["train_val_category_balance"]

    # ── 流水线数据卡片 ──
    flow_cards = f"""
      <div class="stat-card">
        <div class="stat-num">{ps['raw_total']}</div>
        <div class="stat-label">原始数据</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="stat-card warn">
        <div class="stat-num">−{abs(ps.get('quality_filter_removed', 0))}</div>
        <div class="stat-label">质量过滤</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="stat-card ok">
        <div class="stat-num">+{ps['noise_token_fixed']}</div>
        <div class="stat-label">噪声修复</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="stat-card warn">
        <div class="stat-num">−{ps['exact_dedup_removed'] + ps['fuzzy_dedup_removed']}</div>
        <div class="stat-label">去重移除</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="stat-card highlight">
        <div class="stat-num">{ps['after_dedup']}</div>
        <div class="stat-label">最终数据</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="stat-card train">
        <div class="stat-num">{ps['train_size']}</div>
        <div class="stat-label">训练集</div>
      </div>
      <div class="stat-card val">
        <div class="stat-num">{ps['val_size']}</div>
        <div class="stat-label">验证集</div>
      </div>
    """

    # ── 类别分布 Chart.js 数据 ──
    cat_labels = list(cat_dist.keys())
    cat_values = list(cat_dist.values())
    cat_colors = [
        "#e07b39", "#5b8dee", "#50c878", "#e05c7b",
        "#f5c518", "#9b59b6", "#1abc9c", "#e74c3c",
    ]

    # ── 动词 TOP 分布 ──
    verb_labels = list(verb_dist.keys())
    verb_values = list(verb_dist.values())

    # ── 工具函数分布（TOP 10）──
    tool_items = list(tool_dist.items())[:10]
    tool_labels = [t[0] for t in tool_items]
    tool_values = [t[1] for t in tool_items]

    # ── 分层平衡表 ──
    balance_rows = ""
    for cat, info in balance.items():
        total = info["train"] + info["val"]
        ratio_pct = round(info["val_ratio"] * 100, 1)
        bar_w = int(ratio_pct * 6)
        balance_rows += f"""
        <tr>
          <td><span class="cat-badge">{cat}</span></td>
          <td>{total}</td>
          <td>{info['train']}</td>
          <td>{info['val']}</td>
          <td>
            <div class="ratio-bar-bg">
              <div class="ratio-bar" style="width:{bar_w}px"></div>
            </div>
            {ratio_pct}%
          </td>
        </tr>"""

    # ── 长度统计卡片 ──
    len_cards = f"""
      <div class="len-card"><span class="len-val">{len_stats['min']}</span><span class="len-lbl">最短</span></div>
      <div class="len-card"><span class="len-val">{len_stats['max']}</span><span class="len-lbl">最长</span></div>
      <div class="len-card"><span class="len-val">{len_stats['mean']}</span><span class="len-lbl">均值</span></div>
      <div class="len-card"><span class="len-val">{len_stats['median']}</span><span class="len-lbl">中位</span></div>
      <div class="len-card"><span class="len-val">{len_stats['p10']}</span><span class="len-lbl">P10</span></div>
      <div class="len-card"><span class="len-val">{len_stats['p90']}</span><span class="len-lbl">P90</span></div>
    """

    # ── 评估报告部分（可选）──
    eval_section = ""
    if eval_report:
        eval_rows = ""
        headers = [r["label"] for r in eval_report]
        metrics = [
            ("三段式完整率 ★",   lambda r: f"{r['structural_validity']['fully_valid_rate']}%"),
            ("含 &lt;think&gt;",  lambda r: f"{r['structural_validity']['has_think_rate']}%"),
            ("含工具调用",        lambda r: f"{r['structural_validity']['has_tool_code_rate']}%"),
            ("工具调用语法合法率 ★", lambda r: f"{r['tool_call_validity']['syntax_valid_rate']}%"),
            ("风格达标率 ★",      lambda r: f"{r['style_consistency']['passes_threshold_rate']}%"),
            ("关键词密度/百字",   lambda r: str(r['style_consistency']['avg_keyword_density'])),
            ("平均输出长度",      lambda r: str(r['output_length']['mean'])),
        ]
        # 表头
        th_row = "<tr><th>指标</th>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        for label, fn in metrics:
            row = f"<tr><td>{label}</td>"
            vals = []
            for r in eval_report:
                try:
                    vals.append(fn(r))
                except (KeyError, TypeError):
                    vals.append("N/A")
            # 高亮最大值（数字类）
            numeric_vals = []
            for v in vals:
                try:
                    numeric_vals.append(float(v.replace("%", "")))
                except ValueError:
                    numeric_vals.append(None)
            max_v = max((v for v in numeric_vals if v is not None), default=None)
            for v, nv in zip(vals, numeric_vals):
                cls = " class='best'" if nv is not None and nv == max_v and max_v > 0 else ""
                row += f"<td{cls}>{v}</td>"
            row += "</tr>"
            eval_rows += row

        eval_section = f"""
    <section class="section">
      <h2>📊 模型评估报告</h2>
      <table class="eval-table">
        {th_row}
        {eval_rows}
      </table>
      <p class="tip">★ 为核心指标 &nbsp;|&nbsp; <span class="best-sample">绿色</span> 为最优值</p>
    </section>"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>谋杀湾·数据可视化报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d0f14;
    --surface: #161b27;
    --surface2: #1e2535;
    --border: #2a3347;
    --accent: #c8aa6e;
    --accent2: #5b8dee;
    --text: #cdd5e0;
    --dim: #6b7a99;
    --ok: #50c878;
    --warn: #f0a500;
    --danger: #e05c7b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}
  .header {{
    background: linear-gradient(135deg, #0d0f14 0%, #1a1033 50%, #0d1420 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 48px 32px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(200,170,110,0.06) 0%, transparent 70%);
  }}
  .header h1 {{
    font-size: 28px; font-weight: 700;
    color: var(--accent);
    letter-spacing: 3px;
    text-shadow: 0 0 30px rgba(200,170,110,0.3);
  }}
  .header .subtitle {{
    color: var(--dim); margin-top: 8px; font-size: 13px;
    letter-spacing: 1px;
  }}
  .header .meta {{
    position: absolute; right: 48px; top: 40px;
    color: var(--dim); font-size: 12px; text-align: right;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
  .section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
  }}
  .section h2 {{
    font-size: 16px; font-weight: 600;
    color: var(--accent); margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    letter-spacing: 1px;
  }}

  /* 流水线卡片 */
  .flow-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
  .stat-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px; text-align: center; min-width: 90px;
  }}
  .stat-card.warn  {{ border-color: var(--warn);   background: rgba(240,165,0,0.06); }}
  .stat-card.ok    {{ border-color: var(--ok);     background: rgba(80,200,120,0.06); }}
  .stat-card.highlight {{ border-color: var(--accent); background: rgba(200,170,110,0.08); }}
  .stat-card.train {{ border-color: var(--accent2); background: rgba(91,141,238,0.07); }}
  .stat-card.val   {{ border-color: #9b59b6;       background: rgba(155,89,182,0.07); }}
  .stat-num {{ font-size: 26px; font-weight: 700; color: #fff; }}
  .stat-card.warn  .stat-num {{ color: var(--warn); }}
  .stat-card.ok    .stat-num {{ color: var(--ok); }}
  .stat-card.highlight .stat-num {{ color: var(--accent); }}
  .stat-card.train .stat-num {{ color: var(--accent2); }}
  .stat-card.val   .stat-num {{ color: #b47fd4; }}
  .stat-label {{ font-size: 12px; color: var(--dim); margin-top: 4px; }}
  .flow-arrow {{ color: var(--dim); font-size: 20px; padding: 0 4px; }}

  /* 图表网格 */
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .chart-box {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 20px;
  }}
  .chart-box h3 {{ font-size: 13px; color: var(--dim); margin-bottom: 16px; }}
  .chart-box canvas {{ max-height: 280px; }}

  /* 分层平衡表 */
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--dim); font-size: 12px; font-weight: 500; text-transform: uppercase; }}
  td {{ font-size: 13px; }}
  .cat-badge {{
    background: rgba(200,170,110,0.12);
    border: 1px solid rgba(200,170,110,0.3);
    border-radius: 4px; padding: 2px 8px;
    color: var(--accent); font-size: 11px; font-weight: 600;
  }}
  .ratio-bar-bg {{
    display: inline-block; width: 60px; height: 6px;
    background: var(--border); border-radius: 3px; vertical-align: middle; margin-right: 8px;
  }}
  .ratio-bar {{ height: 6px; background: var(--accent2); border-radius: 3px; }}

  /* 长度统计 */
  .len-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .len-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 20px; text-align: center; min-width: 80px;
    display: flex; flex-direction: column; gap: 4px;
  }}
  .len-val {{ font-size: 22px; font-weight: 700; color: var(--accent2); }}
  .len-lbl {{ font-size: 11px; color: var(--dim); }}

  /* 评估表 */
  .eval-table th {{ background: var(--surface); position: sticky; top: 0; }}
  .eval-table td.best {{ color: var(--ok); font-weight: 600; }}
  .best-sample {{ color: var(--ok); }}
  .tip {{ font-size: 12px; color: var(--dim); margin-top: 12px; }}

  /* 全宽图表 */
  .chart-full {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
  .chart-full h3 {{ font-size: 13px; color: var(--dim); margin-bottom: 16px; }}
  .chart-full canvas {{ max-height: 240px; }}
</style>
</head>
<body>

<div class="header">
  <h1>⚓ 谋杀湾·数据可视化报告</h1>
  <div class="subtitle">Murder Bay · Narrative Agent · LLM Fine-tuning Pipeline</div>
  <div class="meta">生成时间：{now}<br>基座模型：Qwen2.5-7B-Instruct + LoRA</div>
</div>

<div class="container">

  <!-- 数据流水线摘要 -->
  <section class="section">
    <h2>⚙️ 数据处理流水线</h2>
    <div class="flow-row">
      {flow_cards}
    </div>
  </section>

  <!-- 图表网格 -->
  <section class="section">
    <h2>📈 数据分布分析</h2>
    <div class="charts-grid">

      <div class="chart-box">
        <h3>行为类别分布（8类）</h3>
        <canvas id="catChart"></canvas>
      </div>

      <div class="chart-box">
        <h3>工具函数调用频次 TOP 10</h3>
        <canvas id="toolChart"></canvas>
      </div>

    </div>

    <div style="margin-top:24px" class="chart-full">
      <h3>动词分布 TOP 30</h3>
      <canvas id="verbChart"></canvas>
    </div>
  </section>

  <!-- 输出长度统计 -->
  <section class="section">
    <h2>📏 输出长度分布</h2>
    <div class="len-row">
      {len_cards}
    </div>
  </section>

  <!-- 分层平衡 -->
  <section class="section">
    <h2>⚖️ 训练 / 验证集分层平衡</h2>
    <table>
      <tr>
        <th>类别</th><th>总计</th><th>训练集</th><th>验证集</th><th>验证比例</th>
      </tr>
      {balance_rows}
    </table>
  </section>

  {eval_section}

</div>

<script>
const DARK_GRID = {{ color: 'rgba(255,255,255,0.05)' }};
const DARK_TICK = {{ color: '#6b7a99' }};
const baseOpts = {{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#cdd5e0', font: {{ size: 12 }} }} }} }},
}};

// ── 类别饼图 ──
new Chart(document.getElementById('catChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(cat_labels, ensure_ascii=False)},
    datasets: [{{
      data: {json.dumps(cat_values)},
      backgroundColor: {json.dumps(cat_colors)},
      borderColor: '#161b27', borderWidth: 2,
    }}]
  }},
  options: {{
    ...baseOpts,
    cutout: '55%',
    plugins: {{
      legend: {{ position: 'right', labels: {{ color: '#cdd5e0', padding: 12, font: {{ size: 12 }} }} }}
    }}
  }}
}});

// ── 工具函数条形图 ──
new Chart(document.getElementById('toolChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(tool_labels, ensure_ascii=False)},
    datasets: [{{
      label: '调用次数',
      data: {json.dumps(tool_values)},
      backgroundColor: 'rgba(91,141,238,0.7)',
      borderColor: 'rgba(91,141,238,1)',
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    ...baseOpts,
    indexAxis: 'y',
    scales: {{
      x: {{ grid: DARK_GRID, ticks: DARK_TICK }},
      y: {{ grid: {{ display: false }}, ticks: {{ ...DARK_TICK, font: {{ size: 11 }} }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

// ── 动词条形图 ──
new Chart(document.getElementById('verbChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(verb_labels, ensure_ascii=False)},
    datasets: [{{
      label: '样本数',
      data: {json.dumps(verb_values)},
      backgroundColor: 'rgba(200,170,110,0.6)',
      borderColor: 'rgba(200,170,110,0.9)',
      borderWidth: 1,
      borderRadius: 3,
    }}]
  }},
  options: {{
    ...baseOpts,
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ ...DARK_TICK, font: {{ size: 11 }} }} }},
      y: {{ grid: DARK_GRID, ticks: DARK_TICK }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});
</script>

</body>
</html>
"""


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成谋杀湾数据可视化 HTML 报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-report", default="dataset/data_report.json")
    parser.add_argument("--eval-report", default="dataset/eval_report.json")
    parser.add_argument("--output",      default="reports/murder_bay_report.html")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    os.chdir(root)

    data_report = load_json(args.data_report)
    if not data_report:
        print(f"[错误] 找不到数据报告: {args.data_report}")
        print("       请先运行: python scripts/data_pipeline.py")
        return

    eval_report = load_json(args.eval_report)
    if eval_report:
        print(f"[信息] 已加载评估报告: {args.eval_report}")
    else:
        print(f"[信息] 未找到评估报告 ({args.eval_report})，将跳过评估部分")
        print("       如需评估，请先运行: python scripts/eval.py")

    html = build_html(data_report, eval_report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ 报告已生成: {out_path.resolve()}")
    print("  用浏览器打开即可查看（需要网络加载 Chart.js）")


if __name__ == "__main__":
    main()
