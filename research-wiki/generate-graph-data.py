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
    """Restore MathJax blocks from placeholders."""
    for i, m in enumerate(blocks):
        text = text.replace(f'\x00MATH{i}\x00', m)
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

        # Regular paragraph — collect consecutive non-empty, non-list lines
        para_lines = []
        while i < len(lines) and lines[i].strip() != '' \
                and not re.match(r'^(\s*)[-*]\s+', lines[i].strip()) \
                and not re.match(r'^(\s*)\d+\.\s+', lines[i].strip()):
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
        papers.append({
            'id': fm.get('node_id', f'paper:{slug}'),
            'slug': slug,
            'short': fm.get('short', infer_short(slug)),
            'title': fm.get('title', slug),
            'authors': authors if isinstance(authors, list) else [authors],
            'year': fm.get('year'),
            'venue': fm.get('venue', ''),
            'arxiv': arxiv,
            'tags': tags if isinstance(tags, list) else [],
            'nodeColor': slug_color(slug),
            'thesis': sections.get('One-line thesis', '').strip(),
            'sections': {k: render_markdown_to_html(embed_section_images(v, md_file)) for k, v in sections.items() if k != 'Connections'},
        })


    papers.sort(key=lambda p: (p['year'] or 9999, p['slug']))

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
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh}
.header{background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);border-bottom:1px solid #30363d;padding:32px 40px 24px}
.header h1{font-size:28px;font-weight:700;background:linear-gradient(90deg,#79c0ff,#3fb950,#f0883e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.header .subtitle{color:#8b949e;font-size:14px;margin-top:6px}
.stats-bar{display:flex;gap:24px;margin-top:16px;flex-wrap:wrap}
.stat{display:flex;align-items:center;gap:8px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 16px;font-size:13px}
.stat .num{font-weight:700;font-size:18px;color:#f0f6fc}
.stat .label{color:#8b949e}
.stat .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.search-bar{display:flex;align-items:center;gap:8px;margin-top:16px;max-width:420px}
.search-bar input{flex:1;background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:9px 14px;font-size:13px;color:#f0f6fc;outline:none;transition:border-color .2s}
.search-bar input:focus{border-color:#79c0ff}
.search-bar input::placeholder{color:#484f58}
.search-bar .search-count{font-size:12px;color:#8b949e;white-space:nowrap;min-width:50px}
.main{display:flex;min-height:calc(100vh - 160px)}
.graph-panel{flex:1;position:relative;min-height:500px}
.graph-panel svg{width:100%;height:100%;display:block}
.graph-title{position:absolute;top:16px;left:24px;font-size:16px;font-weight:600;color:#f0f6fc;pointer-events:none}
.graph-legend{position:absolute;bottom:20px;left:24px;background:rgba(22,27,34,0.92);border:1px solid #30363d;border-radius:8px;padding:12px 16px;font-size:12px;pointer-events:none}
.graph-legend div{margin:4px 0;display:flex;align-items:center;gap:8px}
.graph-legend .swatch{width:28px;height:3px;border-radius:2px}
.side-panel{width:380px;border-left:1px solid #30363d;background:#161b22;overflow-y:auto;max-height:calc(100vh - 160px)}
.side-panel::-webkit-scrollbar{width:6px}
.side-panel::-webkit-scrollbar-track{background:#161b22}
.side-panel::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
.node-card{padding:20px;border-bottom:1px solid #21262d;cursor:pointer;transition:background .15s}
.node-card:hover{background:#1c2128}
.node-card h3{font-size:15px;font-weight:600;color:#79c0ff;margin-bottom:4px;line-height:1.4}
.node-card .meta{font-size:12px;color:#8b949e;margin-bottom:6px}
.node-card .thesis{font-size:13px;color:#c9d1d9;line-height:1.5;margin-bottom:8px}
.node-card .tags{display:flex;flex-wrap:wrap;gap:4px}
.node-card .tag{font-size:11px;padding:2px 8px;border-radius:12px;border:1px solid}
.node-card .conn{margin-top:10px;font-size:12px;color:#8b949e;border-top:1px solid #21262d;padding-top:8px}
.node-card .conn div{margin:3px 0}
.node-card .conn .ext{color:#3fb950}
.node-card .conn .extby{color:#79c0ff}
.tooltip{position:absolute;background:rgba(22,27,34,0.95);border:1px solid #30363d;border-radius:8px;padding:12px 16px;font-size:13px;pointer-events:none;max-width:300px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,0.4);opacity:0;transition:opacity .15s;z-index:100}
.tooltip.visible{opacity:1}
.tooltip strong{color:#f0f6fc}
.nodes circle{stroke:#0d1117;stroke-width:3px;cursor:pointer;transition:stroke-width .15s}
.nodes circle:hover{stroke-width:5px}
.labels text{font-size:13px;font-weight:600;pointer-events:none}
.labels .sub{font-size:11px;fill:#8b949e;font-weight:400}
.edges line{stroke-opacity:.6}
.detail-modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);z-index:1000;justify-content:center;align-items:center}
.detail-modal.open{display:flex}
.detail-content{background:#161b22;border:1px solid #30363d;border-radius:12px;max-width:700px;width:90%;max-height:85vh;overflow-y:auto;padding:32px;position:relative}
.detail-close{position:absolute;top:16px;right:20px;background:none;border:none;color:#8b949e;font-size:24px;cursor:pointer}
.detail-close:hover{color:#f0f6fc}
.detail-content h2{font-size:20px;color:#f0f6fc;margin-bottom:4px}
.detail-content .meta-line{font-size:13px;color:#8b949e;margin-bottom:16px}
.detail-content section{margin-bottom:20px}
.detail-content section h3{font-size:14px;font-weight:600;color:#79c0ff;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #21262d}
.detail-content section p,.detail-content section li{font-size:13px;line-height:1.6;color:#c9d1d9}
.detail-content ul{padding-left:18px}
.detail-content li{margin-bottom:4px}
.detail-content .conn-detail{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:12px;margin-top:8px}
.detail-content .conn-detail div{margin:4px 0;font-size:13px}
@media(max-width:800px){.main{flex-direction:column}.side-panel{width:100%;border-left:none;border-top:1px solid #30363d;max-height:none}.graph-panel{min-height:400px}.header{padding:24px 20px}}
</style>
</head>
<body>
<div class="header">
  <h1>VLM Research Knowledge Graph</h1>
  <div class="subtitle" id="headerSubtitle">Loading...</div>
  <div class="stats-bar" id="statsBar"></div>
  <div class="search-bar"><input type="text" id="searchInput" placeholder="Search papers, authors, tags, content..." autofocus><span class="search-count" id="searchCount"></span></div>
</div>
<div class="main">
  <div class="graph-panel" id="graph">
    <div class="graph-title">Knowledge Graph · Force-Directed</div>
    <div class="graph-legend"><div><span class="swatch" style="background:#79c0ff"></span> extends / builds upon</div></div>
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
    var m=[a,p.venue,p.year].filter(Boolean).join(' · ')+((p.arxiv)?' · arXiv:'+p.arxiv:'');
    var tg=p.tags.map(function(t){return'<span class="tag" style="background:'+p.nodeColor+'18;color:'+p.nodeColor+';border-color:'+p.nodeColor+'33">'+t+'</span>'}).join('');
    var cn=p.connections.map(function(c){return'<div class="'+(c.type==='extends'?'ext':'extby')+'">'+(c.type==='extends'?'→ Extends':'← Extended by')+': '+c.target+'</div>'}).join('');
    c.innerHTML='<h3>'+p.short+': '+p.title.replace(/:/g,':<wbr>')+'</h3><div class="meta">'+m+'</div><div class="thesis">'+p.thesis+'</div><div class="tags">'+tg+'</div><div class="conn">'+cn+'</div>';
    c.addEventListener('click',function(){openDetail(p.id)});
    side.appendChild(c);
  });

  // Search
  var si=document.getElementById('searchInput');
  si.addEventListener('input',function(){searchPapers(this.value)});
  searchPapers(si.value);

  // D3 force graph
  var el=document.getElementById('graph');
  var w=el.clientWidth,h=Math.max(500,window.innerHeight-200);
  var svg=d3.select('#graph').append('svg').attr('width',w).attr('height',h);
  var g=svg.append('g');
  d3.zoom().scaleExtent([0.3,3]).on('zoom',function(e){g.attr('transform',e.transform)})(svg);

  var nd=PAPERS.map(function(p){return Object.assign({},p)});
  var ld=EDGES.map(function(e){return{source:nd.findIndex(function(n){return n.id===e.from}),target:nd.findIndex(function(n){return n.id===e.to}),evidence:e.evidence}}).filter(function(d){return d.source>=0&&d.target>=0});

  var sim=d3.forceSimulation(nd).force('link',d3.forceLink(ld).distance(200).strength(0.5)).force('charge',d3.forceManyBody().strength(-600)).force('center',d3.forceCenter(w/2,h/2)).force('collision',d3.forceCollide().radius(100));

  var defs=svg.append('defs');
  defs.append('marker').attr('id','arrow').attr('viewBox','0 -5 10 10').attr('refX',48).attr('refY',0).attr('markerWidth',8).attr('markerHeight',8).attr('orient','auto').append('path').attr('d','M0,-4L10,0L0,4').attr('fill','#79c0ff').attr('opacity',0.6);

  var link=g.append('g').selectAll('line').data(ld).join('line').attr('stroke','#79c0ff').attr('stroke-width',2.5).attr('stroke-dasharray','6,3').attr('marker-end','url(#arrow)');
  g.append('g').selectAll('text').data(ld).join('text').attr('font-size',10).attr('fill','#8b949e').attr('text-anchor','middle').attr('dy',-6).text('extends');

  var node=g.append('g').selectAll('g').data(nd).join('g').call(d3.drag().on('start',function(e,d){if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y}).on('drag',function(e,d){d.fx=e.x;d.fy=e.y}).on('end',function(e,d){if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));
  node.append('circle').attr('r',42).attr('fill',function(d){return d.nodeColor}).attr('opacity',0.85).on('mouseover',function(e,d){showTooltip(e,d)}).on('mouseout',function(){hideTooltip()}).on('click',function(e,d){e.stopPropagation();openDetail(d.id)});
  node.append('text').attr('dy',-6).attr('fill','#f0f6fc').attr('text-anchor','middle').attr('pointer-events','none').text(function(d){return d.short});
  node.append('text').attr('dy',10).attr('class','sub').attr('text-anchor','middle').attr('pointer-events','none').text(function(d){return d.year});
  window._nodeSel=node;


  sim.on('tick',function(){
    link.attr('x1',function(d){return d.source.x}).attr('y1',function(d){return d.source.y}).attr('x2',function(d){return d.target.x}).attr('y2',function(d){return d.target.y});
    
    node.attr('transform',function(d){return 'translate('+d.x+','+d.y+')'});
  });
  window.addEventListener('resize',function(){var cw=el.clientWidth,ch=Math.max(500,window.innerHeight-200);svg.attr('width',cw).attr('height',ch);sim.force('center',d3.forceCenter(cw/2,ch/2)).alpha(0.3).restart()});
}

function searchPapers(query){
  var q=query.toLowerCase().trim();
  var cards=document.getElementById('sidePanel').children;
  var count=PAPERS.length;
  var matches={};
  if(q){
    count=0;
    PAPERS.forEach(function(p,i){
      var haystack=[p.title,p.short,p.authors.join(' '),p.tags.join(' '),p.thesis,p.venue,String(p.year||''),p.slug];
      var match=haystack.join(' ').toLowerCase().indexOf(q)!==-1;
      matches[p.id]=match;
      if(match)count++;
      if(cards[i])cards[i].style.display=match?'':'none';
    });
  }else{PAPERS.forEach(function(p,i){matches[p.id]=true;if(cards[i])cards[i].style.display=''})}
  document.getElementById('searchCount').textContent=q?(count+'/'+PAPERS.length+' results'):'';
  if(window._nodeSel){
    window._nodeSel.select('circle').attr('opacity',function(d){return matches[d.id]?0.85:0.15});
    window._nodeSel.selectAll('text').attr('opacity',function(d){return matches[d.id]?1:0.15});
  }
}
function showTooltip(e,d){
  var tip=document.getElementById('tooltip');
  var ex=d.connections.filter(function(c){return c.type==='extends'}).map(function(c){return c.target}).join(', ');
  var exBy=d.connections.filter(function(c){return c.type==='extended_by'}).map(function(c){return c.target}).join(', ');
  var a=d.authors.length?d.authors[0]+' et al.':'';
  var ar=d.arxiv?' · arXiv:'+d.arxiv:'';
  tip.innerHTML='<strong>'+d.short+': '+d.title+'</strong><br>'+a+' · '+d.venue+' '+d.year+ar+'<br><br>'+d.thesis+'<br><br>'+(ex?'→ Extends: '+ex+'<br>':'')+(exBy?'← Extended by: '+exBy:'');
  var r=document.getElementById('graph').getBoundingClientRect();
  tip.style.left=(e.pageX-r.left+12)+'px';tip.style.top=(e.pageY-r.top-10)+'px';
  tip.classList.add('visible');
}
function hideTooltip(){document.getElementById('tooltip').classList.remove('visible')}

function openDetail(id){
  var p=PAPERS.find(function(x){return x.id===id});if(!p)return;
  var body=document.getElementById('detailBody');
  var a=p.authors.join(', ');
  var ar=p.arxiv?'· <a href="https://arxiv.org/abs/'+p.arxiv+'" target="_blank" style="color:#79c0ff">arXiv:'+p.arxiv+'</a>':'';
  var cn='';
  p.connections.forEach(function(c){cn+='<div>'+(c.type==='extends'?'Extends':'Extended by')+': <strong>'+c.target+'</strong> — '+c.text+'</div>'});
  function slugify(s){return s.replace(/[^\w一-鿿]+/g,'-').replace(/^-|-$/g,'')}
  var sh='',exKeys=['One-line thesis','Connections','Claims','Relevance to This Project'];
  Object.keys(p.sections||{}).forEach(function(k){
    if(exKeys.indexOf(k)!==-1)return;var v=p.sections[k];if(!v||!v.trim())return;
    sh+='<section id="sec-'+slugify(k)+'"><h3>'+k+'</h3>'+v+'</section>'
  });
  body.innerHTML='<h2>'+p.short+': '+p.title+'</h2><div class="meta-line">'+a+' · '+p.venue+' '+p.year+' '+ar+'</div><section id="sec-thesis"><h3>One-line Thesis</h3><p>'+p.thesis+'</p></section>'+sh+'<section id="sec-connections"><h3>Knowledge Graph Connections</h3><div class="conn-detail">'+(cn||'(none)')+'</div></section>';
  body.style.scrollBehavior='smooth';
  document.getElementById('detailModal').classList.add('open');
  (function r(n){if(window.MathJax&&MathJax.typesetPromise){MathJax.typesetPromise([document.getElementById('detailBody')]).catch(console.error)}else if(n>0){setTimeout(function(){r(n-1)},200)}})(10);
}
function closeDetail(){document.getElementById('detailModal').classList.remove('open')}
document.getElementById('detailModal').addEventListener('click',function(e){if(e.target===e.currentTarget)closeDetail()});

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
