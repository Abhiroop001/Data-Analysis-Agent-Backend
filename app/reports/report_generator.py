# app/reports/report_generator.py
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import json

TEMPLATE_STR = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Auto EDA Report</title></head>
<body>
<h1>Automated EDA Report</h1>
<h2>Summary</h2>
<ul>
{% for i in insights %}
  <li>{{ i }}</li>
{% endfor %}
</ul>

<h2>ScaleDown / Compression</h2>
<pre>{{ scaledown | tojson(indent=2) }}</pre>

<h2>Profile</h2>
<p>Profile saved at: {{ profile_html }}</p>

<h2>Anomalies</h2>
<pre>{{ anomalies | tojson(indent=2) }}</pre>

{% if automl %}
<h2>AutoML Recommendation</h2>
<pre>{{ automl | tojson(indent=2) }}</pre>
{% endif %}
</body>
</html>
"""

def generate_report(out_path: str, insights, scaledown, profile_html, anomalies, automl=None):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    env = Environment(autoescape=select_autoescape())
    tmpl = env.from_string(TEMPLATE_STR)
    html = tmpl.render(
        insights=insights,
        scaledown=scaledown,
        profile_html=profile_html,
        anomalies=anomalies,
        automl=automl
    )
    Path(out_path).write_text(html)
    return out_path
