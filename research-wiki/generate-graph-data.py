#!/usr/bin/env python3
"""
Generate knowledge-graph.html from research-wiki markdown files and edges.
Usage: python3 generate-graph-data.py
       Output: research-wiki/knowledge-graph.html (self-contained)
"""

import base64, json, re, sys
from pathlib import Path

WIKI_ROOT = Path(__file__).parent

def parse_frontmatter(text):
    parts = text.split('---', 2)
    if len(parts) < 3:
        return {}, text.strip()
    yaml_text = parts[1].strip()
    body = parts[2].strip()
    fm = {}
    current_key = None
    current_list = None
    for line in yaml_text.split('\n'):
        list_match = re.match(r'^\s{2}- (.+)', line)
        kv_match = re.match(r'^(\w[\w_]*):\s*(.*)', line)
        nested_match = re.match(r'^\s{2}(\w[\w_]*):\s*(.*)', line)
        if list_match:
            val = list_match.group(1).strip().strip('"')
            if current_list is not None:
                current_list.append(val)
        elif kv_match:
            current_key = kv_match.group(1)
            raw_val = kv_match.group(2).strip()
            if current_key in ('tags', 'authors'):
                current_list = None
                if not raw_val or raw_val == '[]':
                    fm[current_key] = []
                elif raw_val.startswith('['):
                    try:
                        fm[current_key] = json.loads(raw_val)
                    except json.JSONDecodeError:
                        fm[current_key] = [v.strip().strip('"').strip("'") for v in raw_val.strip('[]').split(',') if v.strip()]
                else:
                    fm[current_key] = [v.strip().strip('"') for v in raw_val.split(',')]
            elif current_key == 'external_ids':
                current_list = None
                fm[current_key] = {}
                if raw_val and raw_val != '{}':
                    for part in raw_val.split(','):
                        if ':' in part:
                            k, v = part.split(':', 1)
                            fm[current_key][k.strip()] = v.strip().strip('"')
            else:
                current_list = None
                if raw_val in ('null', '', '~'):
                    fm[current_key] = None
                elif raw_val == 'true':
                    fm[current_key] = True
                elif raw_val == 'false':
                    fm[current_key] = False
                elif raw_val.isdigit():
                    fm[current_key] = int(raw_val)
                else:
                    fm[current_key] = raw_val.strip('"')
        elif nested_match and current_key == 'external_ids':
            k, v = nested_match.group(1), nested_match.group(2).strip().strip('"')
            if v == 'null':
                v = None
            if isinstance(fm.get('external_ids'), dict):
                fm['external_ids'][k] = v
            else:
                fm['external_ids'] = {k: v}
    return fm, body


def extract_sections(body):
    sections = {}
    current_h2 = None
    current_lines = []
    for line in body.split('\n'):
        h2_match = re.match(r'^##\s+(.+)', line)
        if h2_match:
            if current_h2:
                sections[current_h2] = '\n'.join(current_lines).strip()
            current_h2 = h2_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_h2:
        sections[current_h2] = '\n'.join(current_lines).strip()
    return sections


def embed_section_images(text, md_path):
    """Replace markdown image refs ![alt](path) with base64-embedded <img> tags.
    Resolves path relative to the markdown file's directory."""
    def _replace(m):
        alt = m.group(1)
        src = m.group(2)
        img_path = (md_path.parent / src).resolve()
        if img_path.exists():
            ext = img_path.suffix.lower()
            mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif', 'svg': 'image/svg+xml'}
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return f'<img src="data:{mime.get(ext.lstrip("."), "image/png")};base64,{b64}" alt="{alt}" style="max-width:100%;border-radius:8px;margin:12px 0">'
        return m.group(0)
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _replace, text)


def _protect_math(text):
    """Temporarily replace MathJax $...$ and $$...$$ with placeholders."""
    blocks = []
    def _save(m):
        blocks.append(m.group(0))
        return f'\x00MATH{len(blocks) - 1}\x00'
    text = re.sub(r'\$\$[^$]+\$\$|\$[^$]+\$', _save, text)
    return text, blocks


def _restore_math(text, blocks):
    """Restore MathJax blocks from placeholders, HTML-escaping < and >."""
    for i, m in enumerate(blocks):
        # Escape < and > to prevent browsers from parsing them as HTML tags
        # (e.g., t_{<j} would otherwise break MathJax rendering)
        escaped = m.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace(f'\x00MATH{i}\x00', escaped)
    return text


def _render_inline(text):
    """Render inline markdown: bold, italic, code, links. Preserves MathJax."""
    text, math_blocks = _protect_math(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: '<a href="'+m.group(2)+'"' + (' target="_blank"' if not m.group(2).startswith('#') else '') + '>'+m.group(1)+'</a>', text)
    return _restore_math(text, math_blocks)


def render_markdown_to_html(text):
    """Convert basic markdown to HTML: lists, paragraphs, inline formatting."""
    if not text or not text.strip():
        return text

    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Empty line — skip
        if stripped == '':
            i += 1
            continue

        # Multi-line MathJax block: $$ ... $$ (spans multiple lines)
        if stripped == '$$':
            math_lines = ['$$']
            i += 1
            while i < len(lines):
                math_lines.append(lines[i])
                if lines[i].strip() == '$$':
                    i += 1
                    break
                i += 1
            result.append(f'<p>{"<br>".join(math_lines)}</p>')
            continue

        # Heading: ### H3, #### H4, ##### H5, ###### H6
        h_match = re.match(r'^(#{3,6})\s+(.+)$', stripped)
        if h_match:
            lvl = len(h_match.group(1))
            result.append(f'<h{lvl}>{_render_inline(h_match.group(2))}</h{lvl}>')
            i += 1
            continue

        # Unordered list: - item  or  * item (but not **bold**)
        ul_match = re.match(r'^(\s*)[-*]\s+(.+)$', stripped)
        if ul_match and not re.match(r'^\s*\*\s*\*\*', stripped):
            result.append('<ul>')
            while i < len(lines):
                ul_match = re.match(r'^(\s*)[-*]\s+(.+)$', lines[i].strip())
                if not ul_match:
                    break
                result.append(f'<li>{_render_inline(ul_match.group(2))}</li>')
                i += 1
            result.append('</ul>')
            continue

        # Ordered list: 1. item
        ol_match = re.match(r'^(\s*)\d+\.\s+(.+)$', stripped)
        if ol_match:
            result.append('<ol>')
            while i < len(lines):
                ol_match = re.match(r'^(\s*)\d+\.\s+(.+)$', lines[i].strip())
                if not ol_match:
                    break
                result.append(f'<li>{_render_inline(ol_match.group(2))}</li>')
                i += 1
            result.append('</ol>')
            continue

        # Markdown table: | col1 | col2 |
        table_match = re.match(r'^\|.+\|$', stripped)
        if table_match:
            # collect all contiguous table rows
            table_rows = []
            while i < len(lines) and re.match(r'^\|.+\|$', lines[i].strip()):
                table_rows.append(lines[i].strip())
                i += 1

            if len(table_rows) >= 2:
                # parse header
                header_cells = [c.strip() for c in table_rows[0].split('|')[1:-1]]
                # skip separator row (|---|---|)
                data_start = 1
                if re.match(r'^[\|\s\-:]+$', table_rows[1]):
                    data_start = 2

                html = '<table><thead><tr>'
                for h in header_cells:
                    html += f'<th>{_render_inline(h)}</th>'
                html += '</tr></thead><tbody>'
                for row in table_rows[data_start:]:
                    cells = [c.strip() for c in row.split('|')[1:-1]]
                    html += '<tr>'
                    for c in cells:
                        html += f'<td>{_render_inline(c)}</td>'
                    html += '</tr>'
                html += '</tbody></table>'
                result.append(html)
            continue

        # Horizontal rule: ---, ***, ___
        hr_match = re.match(r'^(-{3,}|\*{3,}|_{3,})$', stripped)
        if hr_match:
            result.append('<hr>')
            i += 1
            continue

        # Blockquote: > text (handle multi-line blockquotes)
        bq_match = re.match(r'^>\s?(.*)$', stripped)
        if bq_match:
            bq_lines = []
            while i < len(lines):
                bq_m = re.match(r'^>\s?(.*)$', lines[i].strip())
                if not bq_m:
                    break
                bq_lines.append(_render_inline(bq_m.group(1)))
                i += 1
            bq_content = '<br>'.join(bq_lines) if len(bq_lines) > 1 else bq_lines[0]
            result.append(f'<blockquote><p>{bq_content}</p></blockquote>')
            continue

        # Regular paragraph — collect consecutive non-empty, non-list, non-bq, non-hr lines
        para_lines = []
        while i < len(lines) and lines[i].strip() != '' \
                and not re.match(r'^(\s*)[-*]\s+', lines[i].strip()) \
                and not re.match(r'^(\s*)\d+\.\s+', lines[i].strip()) \
                and not re.match(r'^\|.+\|$', lines[i].strip()) \
                and not re.match(r'^(-{3,}|\*{3,}|_{3,})$', lines[i].strip()) \
                and not re.match(r'^>\s?', lines[i].strip()):
            para_lines.append(_render_inline(lines[i].strip()))
            i += 1

        if para_lines:
            para_html = '<br>'.join(para_lines)
            result.append(f'<p>{para_html}</p>')

    return '\n'.join(result)


SHORT_NAMES = {'vit': 'ViT', 'clip': 'CLIP', 'blip': 'BLIP', 'blip2': 'BLIP-2', 'swin': 'Swin'}
COLORS = ['#79c0ff', '#3fb950', '#f0883e', '#da3633', '#a371f7', '#db6d28', '#238636', '#1f6feb', '#795548']

def slug_color(slug):
    h = sum(ord(c) * (i + 1) for i, c in enumerate(slug))
    return COLORS[h % len(COLORS)]

def infer_short(slug):
    m = re.search(r'\d{4}_(.+)', slug)
    if not m:
        return slug[:12]
    name = m.group(1).lower().replace('_', ' ').replace('-', ' ')
    compact = name.replace(' ', '')
    if compact in SHORT_NAMES:
        return SHORT_NAMES[compact]
    return ' '.join(w.capitalize() for w in name.split())[:15]


def arxiv_to_date(arxiv_id):
    """Convert arxiv ID (YYMM.NNNNN) → YYYY-MM string. Returns None on failure or null."""
    if not arxiv_id:
        return None
    m = re.match(r'^(\d{2})(\d{2})\.\d+$', str(arxiv_id))
    if m:
        return f"20{m.group(1)}-{m.group(2)}"
    return None


def date_to_sort_key(date_str, year):
    """YYYY-MM → sortable integer (months since 2020-01). Falls back to year if no date."""
    if date_str:
        y, m = date_str.split('-')
        return (int(y) - 2020) * 12 + (int(m) - 1)
    if year and isinstance(year, int):
        return (year - 2020) * 12  # default to January of that year
    return 0


def date_idx_to_label(idx):
    """Convert month index back to YYYY-MM display string."""
    y = 2020 + idx // 12
    m = (idx % 12) + 1
    return f"{y}-{m:02d}"


def main():
    papers_dir = WIKI_ROOT / 'papers'
    graph_file = WIKI_ROOT / 'graph' / 'edges.jsonl'
    output_file = WIKI_ROOT / 'knowledge-graph.html'

    # Load papers
    papers = []
    for md_file in sorted(papers_dir.glob('*.md')):
        slug = md_file.stem
        text = md_file.read_text(encoding='utf-8')
        fm, body = parse_frontmatter(text)
        sections = extract_sections(body)
        # Normalize thesis key (handle case variations like "One-line Thesis")
        for k in list(sections.keys()):
            if k.lower().replace('-', '').replace(' ', '') == 'onelinethesis':
                if k != 'One-line thesis':
                    sections['One-line thesis'] = sections.pop(k)
                break
        authors = fm.get('authors', [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(',')]
        tags = fm.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
        arxiv = None
        ext = fm.get('external_ids', {}) or {}
        if isinstance(ext, dict):
            arxiv = ext.get('arxiv')
        if not arxiv:
            arxiv = fm.get('arxiv')
        arxiv_date = arxiv_to_date(arxiv)
        papers.append({
            'id': fm.get('node_id', f'paper:{slug}'),
            'slug': slug,
            'short': fm.get('short', infer_short(slug)),
            'title': fm.get('title', slug),
            'authors': authors if isinstance(authors, list) else [authors],
            'year': fm.get('year'),
            'venue': fm.get('venue', ''),
            'arxiv': arxiv,
            'arxivDate': arxiv_date,
            'dateIdx': date_to_sort_key(arxiv_date, fm.get('year')),
            'tags': tags if isinstance(tags, list) else [],
            'nodeColor': slug_color(slug),
            'thesis': render_markdown_to_html(sections.get('One-line thesis', '').strip()),
            'sections': {k: render_markdown_to_html(embed_section_images(v, md_file)) for k, v in sections.items() if k != 'Connections'},
        })


    papers.sort(key=lambda p: (p['dateIdx'], p['slug']))

    # Load edges
    edges = []
    if graph_file.exists():
        for line in graph_file.read_text(encoding='utf-8').strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    edges.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Build connection lists
    paper_map = {p['id']: p for p in papers}
    for p in papers:
        p['connections'] = []
        p['_extends'] = []
        p['_extendedBy'] = []

    for e in edges:
        src = paper_map.get(e.get('from'))
        dst = paper_map.get(e.get('to'))
        if not src or not dst:
            continue
        if e.get('type') == 'extends':
            src['_extends'].append({'target': dst['short'], 'targetId': dst['id'], 'text': e.get('evidence', '')})
            dst['_extendedBy'].append({'target': src['short'], 'targetId': src['id'], 'text': e.get('evidence', '')})

    for p in papers:
        for c in p['_extends']:
            p['connections'].append({'type': 'extends', 'target': c['target'], 'targetId': c['targetId'], 'text': c['text']})
        for c in p['_extendedBy']:
            p['connections'].append({'type': 'extended_by', 'target': c['target'], 'targetId': c['targetId'], 'text': c['text']})
        p['edgeCount'] = len(p['_extends']) + len(p['_extendedBy'])
        del p['_extends']
        del p['_extendedBy']

    data_json = json.dumps({'papers': papers, 'edges': edges}, indent=2, ensure_ascii=False)

    html_template = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLM Research Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\\(', '\\\)']]},
  startup: {pageReady: function() {return MathJax.startup.defaultPageReady();}}
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"></script>
<style>
:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;--bg4:#0d1117;
  --border:#30363d;--border2:#21262d;
  --text:#c9d1d9;--text2:#8b949e;--text3:#f0f6fc;--text4:#484f58;
  --accent:#79c0ff;--green:#3fb950;--orange:#f0883e;
  --hdr:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
  --tip:rgba(22,27,34,0.95);--nodes:#0d1117;
  --overlay:rgba(0,0,0,0.7);--card-bg:#fffef9;
}
[data-theme="light"]{
  --bg:#faf6ee;--bg2:#fffef9;--bg3:#f5f0e5;--bg4:#fffef9;
  --border:#d4cfc4;--border2:#e5e0d5;
  --text:#2d2d2d;--text2:#6b6b6b;--text3:#1a1a1a;--text4:#999;
  --accent:#3b7abf;--green:#2d8a4e;--orange:#c45a1e;
  --hdr:linear-gradient(135deg,#f5f0e5 0%,#faf6ee 100%);
  --tip:rgba(255,254,249,0.97);--nodes:#faf6ee;
  --overlay:rgba(0,0,0,0.5);
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;transition:background .3s,color .3s}
.header{background:var(--hdr);border-bottom:1px solid var(--border);padding:24px 40px 20px;display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px}
.header-left{flex:1;min-width:280px}
.header h1{font-size:28px;font-weight:700;background:linear-gradient(90deg,var(--accent),var(--green),var(--orange));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.header .subtitle{color:var(--text2);font-size:14px;margin-top:6px}
.theme-btn{flex-shrink:0;background:var(--bg2);border:1px solid var(--border);color:var(--text);padding:8px 16px;border-radius:8px;cursor:pointer;font-size:13px;transition:background .2s;display:flex;align-items:center;gap:6px;white-space:nowrap}
.theme-btn:hover{background:var(--bg3)}
.theme-btn .icon{font-size:16px}
.stats-bar{display:flex;gap:24px;margin-top:16px;flex-wrap:wrap}
.stat{display:flex;align-items:center;gap:8px;background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:8px 16px;font-size:13px}
.stat .num{font-weight:700;font-size:18px;color:var(--text3)}
.stat .label{color:var(--text2)}
.stat .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.search-bar{display:flex;align-items:center;gap:8px;margin-top:16px;max-width:420px}
.search-bar input{flex:1;background:var(--bg4);border:1px solid var(--border);border-radius:8px;padding:9px 14px;font-size:13px;color:var(--text3);outline:none;transition:border-color .2s}
.search-bar input:focus{border-color:var(--accent)}
.search-bar input::placeholder{color:var(--text4)}
.search-bar .search-count{font-size:12px;color:var(--text2);white-space:nowrap;min-width:50px}
.search-wrap{position:relative;max-width:420px}
.suggestions{display:none;position:absolute;z-index:200;top:calc(100% + 6px);left:0;right:0;background:var(--tip);border:1px solid var(--border);border-radius:8px;box-shadow:0 12px 28px rgba(0,0,0,0.28);overflow:hidden}
.suggestions.open{display:block}
.suggestion-item{display:block;width:100%;background:transparent;border:0;border-bottom:1px solid var(--border2);padding:10px 12px;text-align:left;color:var(--text);cursor:pointer}
.suggestion-item:last-child{border-bottom:0}
.suggestion-item:hover,.suggestion-item:focus{background:var(--bg3);outline:none}
.suggestion-item strong{display:block;color:var(--accent);font-size:13px;line-height:1.35}
.suggestion-item span{display:block;color:var(--text2);font-size:12px;margin-top:3px;line-height:1.35}
.empty-state{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:32px;text-align:center;color:var(--text2);font-size:14px;line-height:1.6;pointer-events:none}
.side-empty{padding:24px;color:var(--text2);font-size:13px;line-height:1.6}
.time-filter{display:flex;align-items:center;gap:10px;margin-top:12px;max-width:420px;flex-wrap:wrap}
.time-filter .time-label{font-size:12px;color:var(--text2);white-space:nowrap;font-weight:500}
.time-filter input[type=number]{width:72px;background:var(--bg4);border:1px solid var(--border);border-radius:8px;padding:7px 10px;font-size:13px;color:var(--text3);outline:none;text-align:center;-moz-appearance:textfield;transition:border-color .2s}
.time-filter input[type=number]::-webkit-inner-spin-button,.time-filter input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
.time-filter input[type=number]:focus{border-color:var(--accent)}
.time-filter .time-sep{font-size:12px;color:var(--text4)}
.main{display:flex;min-height:calc(100vh - 160px)}
.graph-panel{flex:1;position:relative;min-height:500px}
.graph-panel svg{width:100%;height:100%;display:block}
.graph-title{position:absolute;top:16px;left:24px;font-size:16px;font-weight:600;color:var(--text3);pointer-events:none}
.graph-legend{position:absolute;bottom:20px;left:24px;background:var(--tip);border:1px solid var(--border);border-radius:8px;padding:12px 16px;font-size:12px;pointer-events:none}
.graph-legend div{margin:4px 0;display:flex;align-items:center;gap:8px}
.graph-legend .swatch{width:28px;height:3px;border-radius:2px}
.side-panel{width:380px;border-left:1px solid var(--border);background:var(--bg2);overflow-y:auto;max-height:calc(100vh - 160px)}
.side-panel::-webkit-scrollbar{width:6px}
.side-panel::-webkit-scrollbar-track{background:var(--bg2)}
.side-panel::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.node-card{padding:20px;border-bottom:1px solid var(--border2);cursor:pointer;transition:background .15s}
.node-card:hover{background:var(--bg3)}
.node-card h3{font-size:15px;font-weight:600;color:var(--accent);margin-bottom:4px;line-height:1.4}
.node-card .meta{font-size:12px;color:var(--text2);margin-bottom:6px}
.node-card .thesis{font-size:13px;color:var(--text);line-height:1.5;margin-bottom:8px}
.node-card .tags{display:flex;flex-wrap:wrap;gap:4px}
.node-card .tag{font-size:11px;padding:2px 8px;border-radius:12px;border:1px solid}
.node-card .conn{margin-top:10px;font-size:12px;color:var(--text2);border-top:1px solid var(--border2);padding-top:8px}
.node-card .conn div{margin:3px 0}
.node-card .conn .ext{color:var(--green)}
.node-card .conn .extby{color:var(--accent)}
.tooltip{position:absolute;background:var(--tip);border:1px solid var(--border);border-radius:8px;padding:12px 16px;font-size:13px;pointer-events:none;max-width:300px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,0.25);opacity:0;transition:opacity .15s;z-index:100;color:var(--text)}
.tooltip.visible{opacity:1}
.tooltip strong{color:var(--text3)}
.nodes circle{stroke:var(--nodes);stroke-width:3px;cursor:pointer;transition:stroke-width .15s}
.nodes circle:hover{stroke-width:5px}
.lbl-name{font-size:13px;font-weight:600;pointer-events:none;fill:var(--text3)}
text.sub{font-size:11px;fill:var(--text2);font-weight:400;pointer-events:none}
.edges line{stroke-opacity:.6}
.detail-modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:var(--overlay);z-index:1000;justify-content:center;align-items:center}
.detail-modal.open{display:flex}
.detail-content{background:var(--bg2);border:1px solid var(--border);border-radius:12px;max-width:900px;width:92%;max-height:90vh;overflow-y:auto;padding:36px;position:relative}
.detail-close{position:absolute;top:16px;right:20px;background:none;border:none;color:var(--text2);font-size:24px;cursor:pointer}
.detail-close:hover{color:var(--text3)}
.detail-content h2{font-size:20px;color:var(--text3);margin-bottom:4px}
.detail-content .meta-line{font-size:13px;color:var(--text2);margin-bottom:16px}
.detail-content blockquote{background:var(--bg);border-left:3px solid var(--accent);margin:12px 0;padding:8px 14px;border-radius:0 6px 6px 0;font-size:13px;line-height:1.6;color:var(--text2)}
.detail-content blockquote p{margin:0}
.detail-content hr{border:none;border-top:1px solid var(--border2);margin:20px 0}
.detail-content section{margin-bottom:20px}
.detail-content section h3{font-size:14px;font-weight:600;color:var(--accent);margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--border2)}
.detail-content section h4{font-size:13px;font-weight:600;color:var(--text3);margin:12px 0 6px}
.detail-content section h5{font-size:13px;font-weight:600;color:var(--text2);margin:10px 0 4px}
.detail-content section h6{font-size:12px;font-weight:600;color:var(--text2);margin:8px 0 4px}
.detail-content section p,.detail-content section li{font-size:13px;line-height:1.6;color:var(--text)}
.detail-content section a{color:var(--accent)}
.detail-content .MathJax{overflow-x:auto;overflow-y:hidden;max-width:100%}
.detail-content mjx-container{overflow-x:auto;max-width:100%}
.detail-content ul{padding-left:18px}
.detail-content li{margin-bottom:4px}
.detail-content table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
.detail-content thead{border-bottom:2px solid var(--border)}
.detail-content th{text-align:left;padding:8px 12px;color:var(--text3);font-weight:600;white-space:nowrap}
.detail-content td{padding:8px 12px;color:var(--text);border-bottom:1px solid var(--border2);vertical-align:top}
.detail-content tbody tr:hover{background:var(--bg3)}
.detail-content .conn-detail{background:var(--bg);border:1px solid var(--border2);border-radius:8px;padding:12px;margin-top:8px}
.detail-content .conn-detail div{margin:4px 0;font-size:13px}
@media(max-width:800px){.main{flex-direction:column}.side-panel{width:100%;border-left:none;border-top:1px solid var(--border);max-height:none}.graph-panel{min-height:400px}.header{padding:16px 20px}}
</style>
</head>
<body>
<div class="header">
  <div class="header-left">
    <h1>VLM Research Knowledge Graph</h1>
    <div class="subtitle" id="headerSubtitle">Loading...</div>
    <div class="stats-bar" id="statsBar"></div>
    <div class="search-wrap"><div class="search-bar"><input type="text" id="searchInput" placeholder="输入文章名，然后从候选中选择" autofocus><span class="search-count" id="searchCount"></span></div><div class="suggestions" id="searchSuggestions"></div></div>
  </div>
  <button class="theme-btn" id="themeToggle" title="Switch theme"><span class="icon">☀️</span> Light</button>
  <a href="https://papernotes.org/" target="_blank" class="theme-btn" title="PaperNotes 论文笔记平台" style="text-decoration:none">📝 PaperNotes</a>
</div>
<div class="main">
  <div class="graph-panel" id="graph">
    <div class="graph-title" id="graphTitle">Knowledge Graph · Select a Paper</div>
    <div class="graph-legend"><div><span class="swatch" style="background:var(--accent)"></span> extends / builds upon</div></div>
    <div class="empty-state" id="graphEmpty">输入文章名字并选择一篇论文后，将展示距离该论文 3 步以内的知识图谱。</div>
  </div>
  <div class="side-panel" id="sidePanel"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<div class="detail-modal" id="detailModal">
  <div class="detail-content"><button class="detail-close" onclick="closeDetail()">&times;</button><div id="detailBody"></div></div>
</div>
<script>
// Data embedded at build time
var PAPERS = PAPERS_PLACEHOLDER;
var EDGES = EDGES_PLACEHOLDER;

function init() {
  var nodeMap = {};
  PAPERS.forEach(function(p){nodeMap[p.id]=p});

  // Build connections from edges
  PAPERS.forEach(function(p){p.connections=[];p._ex=[];p._exBy=[]});
  EDGES.forEach(function(e){
    var src=nodeMap[e.from], dst=nodeMap[e.to];
    if(!src||!dst||e.type!=='extends')return;
    src._ex.push({target:dst.short,text:e.evidence});
    dst._exBy.push({target:src.short,text:e.evidence});
  });
  PAPERS.forEach(function(p){
    p._ex.forEach(function(c){p.connections.push({type:'extends',target:c.target,text:c.text})});
    p._exBy.forEach(function(c){p.connections.push({type:'extended_by',target:c.target,text:c.text})});
    delete p._ex;delete p._exBy;
  });

  // Stats bar
  var subtitle=document.getElementById('headerSubtitle');
  subtitle.textContent='Vision-Language Model 知识图谱 · '+PAPERS.length+' 篇论文 · '+EDGES.length+' 条关系';
  var years=PAPERS.map(function(p){return p.year}).filter(Boolean);
  var yr=years.length?Math.min.apply(null,years)+'–'+Math.max.apply(null,years):'—';
  document.getElementById('statsBar').innerHTML='<div class="stat"><span class="num">'+PAPERS.length+'</span><span class="label">Papers</span></div><div class="stat"><span class="num">'+EDGES.length+'</span><span class="label">Relationships</span></div><div class="stat"><span class="num">'+yr+'</span><span class="label">Timeline</span></div>';

  // Side panel
  var side=document.getElementById('sidePanel');side.innerHTML='';
  PAPERS.forEach(function(p){
    var c=document.createElement('div');c.className='node-card';
    var a=p.authors.length?p.authors[0]+' et al.':'';
    var m=[a,p.venue,p.arxivDate||p.year].filter(Boolean).join(' · ')+((p.arxiv)?' · arXiv:'+p.arxiv:'');
    var tg=p.tags.map(function(t){return'<span class="tag" style="background:'+p.nodeColor+'18;color:'+p.nodeColor+';border-color:'+p.nodeColor+'33">'+t+'</span>'}).join('');
    var cn=p.connections.map(function(c){return'<div class="'+(c.type==='extends'?'ext':'extby')+'">'+(c.type==='extends'?'→ Extends':'← Extended by')+': '+c.target+'</div>'}).join('');
    c.innerHTML='<h3>'+p.short+': '+p.title.replace(/:/g,':<wbr>')+'</h3><div class="meta">'+m+'</div><div class="thesis">'+p.thesis+'</div><div class="tags">'+tg+'</div><div class="conn">'+cn+'</div>';
    c.addEventListener('click',function(){openDetail(p.id)});
    side.appendChild(c);
  });

  // Search
  var si=document.getElementById('searchInput');
  si.addEventListener('input',function(){window._selectedPaperId=null;renderSuggestions();applyFilters()});
  si.addEventListener('focus',function(){renderSuggestions()});
  document.addEventListener('click',function(e){
    if(!e.target.closest('.search-wrap'))closeSuggestions();
  });
  applyFilters();

  // D3 force graph
  var el=document.getElementById('graph');
  var w=el.clientWidth,h=Math.max(500,window.innerHeight-200);
  window._graphWidth=w;
  window._graphHeight=h;
  var svg=d3.select('#graph').append('svg').attr('width',w).attr('height',h);
  var g=svg.append('g');
  d3.zoom().scaleExtent([0.3,3]).on('zoom',function(e){g.attr('transform',e.transform)})(svg);

  var nd=PAPERS.map(function(p){return Object.assign({},p)});
  var ld=EDGES.map(function(e){return{source:nd.findIndex(function(n){return n.id===e.from}),target:nd.findIndex(function(n){return n.id===e.to}),evidence:e.evidence}}).filter(function(d){return d.source>=0&&d.target>=0});
  window._graphNodes=nd;
  window._visibleNodeIds={};
  function isVisibleNode(d){return !!(window._visibleNodeIds&&window._visibleNodeIds[d.id])}
  function isVisibleLink(d){return isVisibleNode(d.source)&&isVisibleNode(d.target)}
  var linkForce=d3.forceLink(ld).distance(function(d){return isVisibleLink(d)?190:0}).strength(function(d){return isVisibleLink(d)?0.7:0});
  var chargeForce=d3.forceManyBody().strength(function(d){return isVisibleNode(d)?-420:0});
  var xForce=d3.forceX(function(){return window._graphWidth/2}).strength(function(d){return d.id===window._selectedPaperId?1.4:(isVisibleNode(d)?0.08:0)});
  var yForce=d3.forceY(function(){return window._graphHeight/2}).strength(function(d){return d.id===window._selectedPaperId?1.4:(isVisibleNode(d)?0.08:0)});
  var collisionForce=d3.forceCollide().radius(function(d){return isVisibleNode(d)?96:0});
  var sim=d3.forceSimulation(nd).force('link',linkForce).force('charge',chargeForce).force('center',d3.forceCenter(w/2,h/2).strength(0.1)).force('x',xForce).force('y',yForce).force('collision',collisionForce);
  window._sim=sim;
  window._refreshGraphForces=function(){
    var activeNodes=nd.filter(function(d){return isVisibleNode(d)});
    var activeLinks=ld.filter(function(d){return isVisibleLink(d)});
    sim.nodes(activeNodes);
    linkForce.links(activeLinks);
    linkForce.distance(function(d){return isVisibleLink(d)?190:0}).strength(function(d){return isVisibleLink(d)?0.7:0});
    chargeForce.strength(function(d){return isVisibleNode(d)?-420:0});
    xForce.strength(function(d){return d.id===window._selectedPaperId?1.4:(isVisibleNode(d)?0.08:0)});
    yForce.strength(function(d){return d.id===window._selectedPaperId?1.4:(isVisibleNode(d)?0.08:0)});
    collisionForce.radius(function(d){return isVisibleNode(d)?96:0});
  };

  function getNodeColors(){
    var isLight=document.documentElement.getAttribute('data-theme')==='light';
    return {
      text: isLight?'#1a1a1a':'#f0f6fc',
      muted: isLight?'#555':'#8b949e',
      nodes: isLight?'#faf6ee':'#0d1117'
    };
  }
  var nc=getNodeColors();
  var clText=nc.text,clMuted=nc.muted;

  var defs=svg.append('defs');
  defs.append('marker').attr('id','arrow').attr('viewBox','0 -5 10 10').attr('refX',48).attr('refY',0).attr('markerWidth',8).attr('markerHeight',8).attr('orient','auto').append('path').attr('d','M0,-4L10,0L0,4').attr('fill','#79c0ff').attr('opacity',0.6);

  var link=g.append('g').attr('class','edges').selectAll('line').data(ld).join('line').attr('stroke','#79c0ff').attr('stroke-width',2.5).attr('stroke-dasharray','6,3').attr('marker-end','url(#arrow)');
  window._linkSel=link;
  // Edge labels removed — kept minimal

  var node=g.append('g').attr('class','nodes').selectAll('g').data(nd).join('g').call(d3.drag().on('start',function(e,d){if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y}).on('drag',function(e,d){d.fx=e.x;d.fy=e.y}).on('end',function(e,d){if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));
  node.append('circle').attr('r',42).attr('fill',function(d){return d.nodeColor}).attr('opacity',0.85).on('mouseover',function(e,d){showTooltip(e,d)}).on('mouseout',function(){hideTooltip()}).on('click',function(e,d){e.stopPropagation();openDetail(d.id)});
  node.append('text').attr('dy',-6).attr('fill',clText).attr('class','lbl-name').attr('text-anchor','middle').attr('pointer-events','none').text(function(d){return d.short});
  node.append('text').attr('dy',10).attr('class','sub').attr('fill',clMuted).attr('text-anchor','middle').attr('pointer-events','none').text(function(d){return d.arxivDate||d.year});
  window._nodeSel=node;
  applyFilters();


  sim.on('tick',function(){
    link.attr('x1',function(d){return d.source.x}).attr('y1',function(d){return d.source.y}).attr('x2',function(d){return d.target.x}).attr('y2',function(d){return d.target.y});
    
    node.attr('transform',function(d){return 'translate('+d.x+','+d.y+')'});
  });
  window.addEventListener('resize',function(){
    var cw=el.clientWidth,ch=Math.max(500,window.innerHeight-200);
    window._graphWidth=cw;
    window._graphHeight=ch;
    svg.attr('width',cw).attr('height',ch);
    sim.force('center',d3.forceCenter(cw/2,ch/2).strength(0.1));
    centerSelectedNode();
    sim.alpha(0.5).restart();
  });
}

function buildNeighborhood(rootId,maxSteps){
  var adjacency={};
  PAPERS.forEach(function(p){adjacency[p.id]=[]});
  EDGES.forEach(function(e){
    if(!adjacency[e.from]||!adjacency[e.to])return;
    adjacency[e.from].push(e.to);
    adjacency[e.to].push(e.from);
  });
  var visible={},dist={};
  var queue=[rootId];
  visible[rootId]=true;dist[rootId]=0;
  while(queue.length){
    var id=queue.shift();
    if(dist[id]>=maxSteps)continue;
    (adjacency[id]||[]).forEach(function(next){
      if(visible[next])return;
      visible[next]=true;
      dist[next]=dist[id]+1;
      queue.push(next);
    });
  }
  return {visible:visible,dist:dist};
}
function selectPaper(id){
  var p=PAPERS.find(function(x){return x.id===id});if(!p)return;
  window._selectedPaperId=id;
  document.getElementById('searchInput').value=p.title;
  closeSuggestions();
  applyFilters();
}
function renderSuggestions(){
  var q=document.getElementById('searchInput').value.toLowerCase().trim();
  var box=document.getElementById('searchSuggestions');
  if(!q){box.classList.remove('open');box.innerHTML='';return}
  var results=PAPERS.filter(function(p){
    var haystack=[p.title,p.short,p.slug,p.authors.join(' '),p.tags.join(' '),p.venue,String(p.year||'')].join(' ').toLowerCase();
    return haystack.indexOf(q)!==-1;
  }).slice(0,8);
  if(!results.length){
    box.innerHTML='<button class="suggestion-item" type="button" disabled><strong>没有匹配的论文</strong><span>试试论文简称、标题关键词或作者名。</span></button>';
    box.classList.add('open');
    document.getElementById('searchCount').textContent='0 results';
    return;
  }
  box.innerHTML=results.map(function(p){
    var meta=[p.short,p.venue,p.arxivDate||p.year].filter(Boolean).join(' · ');
    return '<button class="suggestion-item" type="button" data-id="'+p.id+'"><strong>'+p.title+'</strong><span>'+meta+'</span></button>';
  }).join('');
  Array.prototype.forEach.call(box.querySelectorAll('.suggestion-item[data-id]'),function(btn){
    btn.addEventListener('click',function(){selectPaper(btn.getAttribute('data-id'))});
  });
  box.classList.add('open');
  document.getElementById('searchCount').textContent=results.length+'/'+PAPERS.length+' choices';
}
function closeSuggestions(){
  var box=document.getElementById('searchSuggestions');
  box.classList.remove('open');
}
function applyFilters(){
  var side=document.getElementById('sidePanel');
  var cards=side.querySelectorAll('.node-card');
  var selectedId=window._selectedPaperId;
  var matches={};
  var dist={};
  if(selectedId){
    var neighborhood=buildNeighborhood(selectedId,3);
    matches=neighborhood.visible;
    dist=neighborhood.dist;
  }else{
    PAPERS.forEach(function(p){matches[p.id]=false});
  }
  var count=Object.keys(matches).filter(function(id){return matches[id]}).length;
  var empty=side.querySelector('.side-empty');
  if(!selectedId&&!empty){
    empty=document.createElement('div');
    empty.className='side-empty';
    empty.textContent='先输入文章名并从候选栏选择一篇论文。右侧会显示该论文 3 步邻域内的文章。';
    side.insertBefore(empty,side.firstChild);
  }
  if(empty)empty.style.display=selectedId?'none':'block';
  PAPERS.forEach(function(p,i){
    if(cards[i])cards[i].style.display=matches[p.id]?'':'none';
    if(cards[i]&&matches[p.id]){
      cards[i].style.order=String(dist[p.id]||0).padStart(2,'0')+p.short;
    }
  });
  var selected=PAPERS.find(function(p){return p.id===selectedId});
  document.getElementById('searchCount').textContent=selectedId?(count+' nodes'):'';
  document.getElementById('graphTitle').textContent=selected?'3-Hop Neighborhood · '+selected.short:'Knowledge Graph · Select a Paper';
  document.getElementById('graphEmpty').style.display=selectedId?'none':'flex';
  window._visibleNodeIds=matches;
  seedVisibleNeighborhood(selectedId,matches,dist);
  centerSelectedNode();
  if(window._refreshGraphForces)window._refreshGraphForces();
  if(window._nodeSel){
    window._nodeSel.select('circle').attr('opacity',function(d){return matches[d.id]?0.85:0});
    window._nodeSel.selectAll('text').attr('opacity',function(d){return matches[d.id]?1:0});
    window._nodeSel.style('pointer-events',function(d){return matches[d.id]?'auto':'none'});
    window._nodeSel.filter(function(d){return d.id===selectedId}).raise();
  }
  if(window._linkSel){
    window._linkSel.attr('opacity',function(d){return matches[d.source.id]&&matches[d.target.id]?0.6:0});
  }
  if(window._sim)window._sim.alpha(selectedId?0.9:0.2).restart();
}
function centerSelectedNode(){
  if(!window._graphNodes)return;
  var cx=(window._graphWidth||0)/2,cy=(window._graphHeight||0)/2;
  window._graphNodes.forEach(function(d){
    if(d.id===window._selectedPaperId){
      d.x=cx;d.y=cy;d.fx=cx;d.fy=cy;d.vx=0;d.vy=0;
    }else{
      d.fx=null;d.fy=null;
      if(!window._visibleNodeIds||!window._visibleNodeIds[d.id]){d.vx=0;d.vy=0}
    }
  });
}
function seedVisibleNeighborhood(selectedId,matches,dist){
  if(!selectedId||!window._graphNodes)return;
  var cx=(window._graphWidth||0)/2,cy=(window._graphHeight||0)/2;
  var rings={1:[],2:[],3:[]};
  window._graphNodes.forEach(function(d){
    if(d.id!==selectedId&&matches[d.id]){
      var step=dist[d.id]||1;
      if(!rings[step])rings[step]=[];
      rings[step].push(d);
    }
  });
  [1,2,3].forEach(function(step){
    var nodes=rings[step]||[];
    var radius=150+(step-1)*115;
    nodes.forEach(function(d,i){
      var angle=(Math.PI*2*i/Math.max(nodes.length,1))-(Math.PI/2)+(step*0.35);
      d.x=cx+Math.cos(angle)*radius;
      d.y=cy+Math.sin(angle)*radius;
      d.vx=0;d.vy=0;d.fx=null;d.fy=null;
    });
  });
}
function showTooltip(e,d){
  var tip=document.getElementById('tooltip');
  var ex=d.connections.filter(function(c){return c.type==='extends'}).map(function(c){return c.target}).join(', ');
  var exBy=d.connections.filter(function(c){return c.type==='extended_by'}).map(function(c){return c.target}).join(', ');
  var a=d.authors.length?d.authors[0]+' et al.':'';
  var ar=d.arxiv?' · arXiv:'+d.arxiv:'';
  var ad=d.arxivDate?' · '+d.arxivDate:'';
  tip.innerHTML='<strong>'+d.short+': '+d.title+'</strong><br>'+a+' · '+d.venue+' '+d.year+ar+ad+'<br><br>'+d.thesis+'<br><br>'+(ex?'→ Extends: '+ex+'<br>':'')+(exBy?'← Extended by: '+exBy:'');
  var r=document.getElementById('graph').getBoundingClientRect();
  tip.style.left=(e.pageX-r.left+12)+'px';tip.style.top=(e.pageY-r.top-10)+'px';
  tip.classList.add('visible');
}
function hideTooltip(){document.getElementById('tooltip').classList.remove('visible')}

function openDetail(id){
  var p=PAPERS.find(function(x){return x.id===id});if(!p)return;
  var body=document.getElementById('detailBody');
  var a=p.authors.join(', ');
  var ar=p.arxiv?'· <a href="https://arxiv.org/abs/'+p.arxiv+'" target="_blank">arXiv:'+p.arxiv+'</a>':'';
  var cn='';
  p.connections.forEach(function(c){cn+='<div>'+(c.type==='extends'?'Extends':'Extended by')+': <strong>'+c.target+'</strong> — '+c.text+'</div>'});
  function slugify(s){return s.replace(/[^\w一-鿿]+/g,'-').replace(/^-|-$/g,'')}
  var sh='',exKeys=['One-line thesis','Connections','Claims','Relevance to This Project'];
  Object.keys(p.sections||{}).forEach(function(k){
    if(exKeys.indexOf(k)!==-1)return;var v=p.sections[k];if(!v||!v.trim())return;
    sh+='<section><h3>'+k+'</h3>'+v+'</section>'
  });
  body.innerHTML='<h2>'+p.short+': '+p.title+'</h2><div class="meta-line">'+a+' · '+p.venue+' '+(p.arxivDate||p.year)+' '+ar+'</div><section><h3>One-line Thesis</h3><p>'+p.thesis+'</p></section>'+sh+'<section><h3>Knowledge Graph Connections</h3><div class="conn-detail">'+(cn||'(none)')+'</div></section>';
  body.style.scrollBehavior='smooth';
  body.onclick=function(e){
    var a=e.target.closest('a[href^="#sec-"]');
    if(!a)return;
    e.preventDefault();
    var target=a.getAttribute('href').replace('#','');
    var h3s=body.querySelectorAll('section h3');
    for(var i=0;i<h3s.length;i++){
      if(slugify(h3s[i].textContent)===target||('sec-'+slugify(h3s[i].textContent))===target){
        h3s[i].parentElement.scrollIntoView({behavior:'smooth',block:'start'});
        return;
      }
    }
  };
  document.getElementById('detailModal').classList.add('open');
  (function r(n){if(window.MathJax&&MathJax.typesetPromise&&MathJax.typesetPromise.apply){MathJax.typesetPromise([document.getElementById('detailBody')]).catch(function(){});}else if(n>0){setTimeout(function(){r(n-1)},200)}})(25);
}
function closeDetail(){document.getElementById('detailModal').classList.remove('open')}
document.getElementById('detailModal').addEventListener('click',function(e){if(e.target===e.currentTarget)closeDetail()});

(function(){
  var saved=localStorage.getItem('vlm-wiki-theme')||'dark';
  document.documentElement.setAttribute('data-theme',saved);
  var btn=document.getElementById('themeToggle');
  updateBtn(saved);
  btn.addEventListener('click',function(){
    var cur=document.documentElement.getAttribute('data-theme');
    var next=cur==='light'?'dark':'light';
    document.documentElement.setAttribute('data-theme',next);
    localStorage.setItem('vlm-wiki-theme',next);
    updateBtn(next);
    updateGraphColors();
  });
  function updateBtn(t){
    btn.innerHTML=t==='light'?'<span class="icon">🌙</span> Dark':'<span class="icon">☀️</span> Light';
    btn.title=t==='light'?'Switch to dark theme':'Switch to light theme';
  }
  window.updateGraphColors=function(){
    var nc=getNodeColors();
    d3.selectAll('.nodes circle').attr('stroke',nc.nodes);
    d3.selectAll('.lbl-name').attr('fill',nc.text);
    d3.selectAll('text.sub').attr('fill',nc.muted);
  };
})();
document.addEventListener('DOMContentLoaded',init);
</script>
</body>
</html>'''

    # Replace placeholders with actual JSON data
    html_output = html_template.replace('PAPERS_PLACEHOLDER', json.dumps(papers, indent=2, ensure_ascii=False))
    html_output = html_output.replace('EDGES_PLACEHOLDER', json.dumps(edges, indent=2, ensure_ascii=False))

    output_file.write_text(html_output, encoding='utf-8')
    print(f"✓ Generated {output_file}")
    print(f"  {len(papers)} papers, {len(edges)} edges, {len(html_output)} bytes")


if __name__ == '__main__':
    main()
