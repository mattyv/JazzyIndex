#!/usr/bin/env python3
"""Generate an HTML gallery page for all index visualization charts."""

import os
from pathlib import Path

def generate_charts_html():
    """Generate HTML page with all charts from index_data directory."""

    # Get all PNG files in index_data directory
    index_data_dir = Path(__file__).parent / 'images' / 'index_data'
    png_files = sorted([f.name for f in index_data_dir.glob('*.png')])

    # Extract unique distributions, sizes, and segment counts for filtering
    distributions = set()
    sizes = set()
    segments = set()

    for filename in png_files:
        # Parse filename: index_{Distribution}_N{Size}_S{Segments}.png
        if filename.startswith('index_') and filename.endswith('.png'):
            parts = filename[6:-4].split('_')  # Remove 'index_' prefix and '.png' suffix
            if len(parts) >= 3:
                dist = parts[0]
                size = parts[1][1:]  # Remove 'N' prefix
                seg = parts[2][1:]   # Remove 'S' prefix
                distributions.add(dist)
                sizes.add(size)
                segments.add(seg)

    distributions = sorted(distributions)
    sizes = sorted(sizes, key=int)
    segments = sorted(segments, key=int)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JazzyIndex Visualization Charts</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}

        h1 {{
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}

        .subtitle {{
            color: #718096;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}

        .filters {{
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #e2e8f0;
        }}

        .filter-group {{
            margin-bottom: 15px;
        }}

        .filter-group:last-child {{
            margin-bottom: 0;
        }}

        .filter-group label {{
            display: block;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 8px;
        }}

        .filter-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .filter-btn {{
            padding: 8px 16px;
            border: 2px solid #cbd5e0;
            background: white;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9em;
            font-weight: 500;
        }}

        .filter-btn:hover {{
            border-color: #667eea;
            background: #edf2f7;
        }}

        .filter-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .stats {{
            background: #edf2f7;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 30px;
            text-align: center;
            color: #4a5568;
            font-weight: 500;
        }}

        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}

        .chart-card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .chart-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }}

        .chart-card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}

        .chart-info {{
            padding: 15px;
            background: #f7fafc;
        }}

        .chart-title {{
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 5px;
        }}

        .chart-meta {{
            font-size: 0.85em;
            color: #718096;
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            animation: fadeIn 0.3s;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .modal-content {{
            margin: 2% auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            animation: zoomIn 0.3s;
        }}

        @keyframes zoomIn {{
            from {{ transform: scale(0.8); }}
            to {{ transform: scale(1); }}
        }}

        .close {{
            position: absolute;
            top: 30px;
            right: 45px;
            color: #f1f1f1;
            font-size: 50px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }}

        .close:hover {{
            color: #bbb;
        }}

        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }}

        .back-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="VISUALIZATIONS.md" class="back-link">← Back to Visualizations Guide</a>

        <h1>📊 Index Visualization Charts</h1>
        <p class="subtitle">Exploring {len(png_files)} index structure visualizations across different distributions and configurations</p>

        <div class="filters">
            <div class="filter-group">
                <label>Distribution:</label>
                <div class="filter-buttons">
                    <button class="filter-btn active" data-filter="distribution" data-value="all">All</button>
                    {''.join(f'<button class="filter-btn" data-filter="distribution" data-value="{d}">{d}</button>' for d in distributions)}
                </div>
            </div>

            <div class="filter-group">
                <label>Dataset Size:</label>
                <div class="filter-buttons">
                    <button class="filter-btn active" data-filter="size" data-value="all">All</button>
                    {''.join(f'<button class="filter-btn" data-filter="size" data-value="{s}">N={s}</button>' for s in sizes)}
                </div>
            </div>

            <div class="filter-group">
                <label>Segments:</label>
                <div class="filter-buttons">
                    <button class="filter-btn active" data-filter="segments" data-value="all">All</button>
                    {''.join(f'<button class="filter-btn" data-filter="segments" data-value="{seg}">S={seg}</button>' for seg in segments)}
                </div>
            </div>
        </div>

        <div class="stats">
            Showing <span id="visible-count">{len(png_files)}</span> of {len(png_files)} charts
        </div>

        <div class="gallery" id="gallery">
"""

    # Add chart cards
    for filename in png_files:
        if filename.startswith('index_') and filename.endswith('.png'):
            # Parse filename: index_{Distribution}_N{Size}_S{Segments}.png
            parts = filename[6:-4].split('_')
            if len(parts) >= 3:
                dist = parts[0]
                size = parts[1][1:]
                seg = parts[2][1:]

                html_content += f"""            <div class="chart-card" data-distribution="{dist}" data-size="{size}" data-segments="{seg}">
                <img src="images/index_data/{filename}" alt="{filename}" onclick="openModal(this.src)">
                <div class="chart-info">
                    <div class="chart-title">{dist} Distribution</div>
                    <div class="chart-meta">Size: {size} | Segments: {seg}</div>
                </div>
            </div>
"""

    html_content += """        </div>
    </div>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        // Filter state
        const filters = {
            distribution: 'all',
            size: 'all',
            segments: 'all'
        };

        // Initialize filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const filterType = this.dataset.filter;
                const value = this.dataset.value;

                // Update active state for this filter group
                this.parentElement.querySelectorAll('.filter-btn').forEach(b => {
                    b.classList.remove('active');
                });
                this.classList.add('active');

                // Update filter state
                filters[filterType] = value;

                // Apply filters
                applyFilters();
            });
        });

        function applyFilters() {
            const cards = document.querySelectorAll('.chart-card');
            let visibleCount = 0;

            cards.forEach(card => {
                const dist = card.dataset.distribution;
                const size = card.dataset.size;
                const seg = card.dataset.segments;

                const matchesDist = filters.distribution === 'all' || dist === filters.distribution;
                const matchesSize = filters.size === 'all' || size === filters.size;
                const matchesSeg = filters.segments === 'all' || seg === filters.segments;

                if (matchesDist && matchesSize && matchesSeg) {
                    card.style.display = 'block';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            });

            document.getElementById('visible-count').textContent = visibleCount;
        }

        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""

    # Write HTML file
    output_file = Path(__file__).parent / 'charts.html'
    output_file.write_text(html_content)
    print(f"Generated {output_file}")
    print(f"Total charts: {len(png_files)}")
    print(f"Distributions: {', '.join(distributions)}")
    print(f"Sizes: {', '.join(sizes)}")
    print(f"Segments: {', '.join(segments)}")

if __name__ == '__main__':
    generate_charts_html()
