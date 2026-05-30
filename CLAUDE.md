<!-- ARIS:BEGIN -->
## Research Wiki

VLM paper knowledge base with interactive graph visualization, deployed via GitHub Pages.

**Live URL:** `https://komh-liu.github.io/vlm_database/research-wiki/knowledge-graph.html`

**Repo:** `https://github.com/Komh-liu/vlm_database.git` (HTTPS, `gh` auth)

### Structure
```
research-wiki/
  papers/            # Paper .md files (YAML frontmatter + sections)
    dosovitskiy2021_vit.md
    li2023_blip2.md
    liu2021_swin.md
    radford2021_clip.md
  graph/edges.jsonl  # Knowledge graph edges (JSONL)
  images/             # Image assets referenced by papers
  generate-graph-data.py  # Build script → generates knowledge-graph.html
  knowledge-graph.html    # Self-contained interactive D3 graph (auto-generated)
  graph-data.js       # Paper+edge data as JS variables (auto-generated)
  update.sh           # One-liner: generate + git add/commit/push
```

### generate-graph-data.py
Build script that reads all `.md` papers + `edges.jsonl`, produces self-contained `knowledge-graph.html`:
- Parses YAML frontmatter and markdown sections from each paper
- Embeds referenced images as base64 (`embed_section_images`)
- Injects paper/edge JSON into the HTML template


### knowledge-graph.html features
- D3 force-directed graph with drag, zoom, tooltips
- Sidebar card list with click-to-expand detail modal
- Real-time search filtering: title, authors, tags, thesis, venue, year — dims non-matching nodes
- MathJax LaTeX rendering in detail modal with retry (up to 2s)
- Base64-embedded images, GitHub-dark theme, mobile responsive

### Workflow for adding content
1. Add paper `.md` to `papers/` (copy existing paper as template for frontmatter)
2. Add images to `images/`, reference in `.md` as `![alt](../images/file.png)`
3. Add edge to `graph/edges.jsonl`
4. Run `python3 generate-graph-data.py`
5. Push to `main` → GitHub Pages auto-deploys

### Gotchas
- GitHub Pages build sometimes gets stuck ("building" with 0 duration) — push an empty commit to retrigger
- Git credential helper must be configured: `gh auth setup-git`

## ARIS Skill Scope
ARIS skills installed in this project: 78 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/liu/GitRepo/Auto-claude-code-research-in-sleep`).
Update with: `bash /home/liu/GitRepo/Auto-claude-code-research-in-sleep/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
