import os
import sys
import smtplib
import argparse
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from jinja2 import Template

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
REPORTS_DIR = BASE_DIR / "reports"
PROC_DIR = BASE_DIR / "data" / "processed"

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Premium HTML Email Template using Jinja2
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bluestock Fintech - Weekly Mutual Fund Performance Report</title>
    <style>
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            background-color: #f7fafc;
            margin: 0;
            padding: 0;
            color: #2d3748;
        }
        .container {
            max-width: 600px;
            margin: 20px auto;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        .header {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            padding: 30px;
            text-align: center;
            color: #ffffff;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .header p {
            margin: 5px 0 0 0;
            font-size: 14px;
            color: #a0aec0;
        }
        .content {
            padding: 30px;
        }
        .meta-summary {
            background-color: #ebf8ff;
            border-left: 4px solid #3182ce;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 25px;
        }
        .meta-summary h3 {
            margin: 0 0 5px 0;
            color: #2b6cb0;
            font-size: 16px;
        }
        .meta-summary p {
            margin: 0;
            font-size: 14px;
            color: #2d3748;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #1a202c;
            border-bottom: 2px solid #edf2f7;
            padding-bottom: 8px;
            margin-bottom: 15px;
            margin-top: 25px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 13px;
        }
        th {
            background-color: #f7fafc;
            color: #4a5568;
            font-weight: 600;
            text-align: left;
            padding: 10px;
            border-bottom: 2px solid #e2e8f0;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #edf2f7;
            color: #4a5568;
        }
        tr:hover {
            background-color: #f8fafc;
        }
        .badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-high { background-color: #fed7d7; color: #9b2c2c; }
        .badge-mod { background-color: #feebc8; color: #9c4221; }
        .badge-low { background-color: #c6f6d5; color: #22543d; }
        .footer {
            background-color: #f7fafc;
            padding: 20px;
            text-align: center;
            font-size: 11px;
            color: #a0aec0;
            border-top: 1px solid #edf2f7;
        }
        .button {
            display: inline-block;
            background-color: #3182ce;
            color: #ffffff !important;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            margin-top: 15px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BLUESTOCK FINTECH</h1>
            <p>Weekly Mutual Fund Performance Summary</p>
        </div>
        <div class="content">
            <div class="meta-summary">
                <h3>Report Insights</h3>
                <p>Generated on: <strong>{{ date_str }}</strong> | Total tracked funds: <strong>{{ total_funds }}</strong></p>
            </div>
            
            <div class="section-title">🏆 Top 5 Funds of the Week (by Composite Score)</div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Scheme Name</th>
                        <th>Category</th>
                        <th>Risk</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for fund in top_funds %}
                    <tr>
                        <td><strong>#{{ fund.rank_overall }}</strong></td>
                        <td>{{ fund.scheme_name }}</td>
                        <td>{{ fund.category }}</td>
                        <td>
                            {% if 'high' in fund.risk_category.lower() %}
                            <span class="badge badge-high">{{ fund.risk_category }}</span>
                            {% elif 'moderate' in fund.risk_category.lower() %}
                            <span class="badge badge-mod">{{ fund.risk_category }}</span>
                            {% else %}
                            <span class="badge badge-low">{{ fund.risk_category }}</span>
                            {% endif %}
                        </td>
                        <td><strong>{{ "%.1f"|format(fund.composite_score) }}</strong></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <div class="section-title">📊 Key Metrics Summary</div>
            <p>Here are the average performance parameters across our fund categories:</p>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Avg 3-Yr CAGR</th>
                        <th>Avg Sharpe Ratio</th>
                        <th>Avg Expense Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cat in cat_stats %}
                    <tr>
                        <td><strong>{{ cat.category }}</strong></td>
                        <td>{{ "%.2f"|format(cat.cagr_3yr_pct) }}%</td>
                        <td>{{ "%.2f"|format(cat.sharpe_ratio) }}</td>
                        <td>{{ "%.2f"|format(cat.expense_ratio_pct) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <center>
                <a href="#" class="button">Go to Analytics Dashboard</a>
            </center>
        </div>
        <div class="footer">
            <p>Sent by Bluestock Fintech Analytics Division.</p>
            <p>This is a simulated performance report for educational/project purposes. Past performance is not an indicator of future results.</p>
        </div>
    </div>
</body>
</html>
"""

def generate_report():
    scorecard_path = PROC_DIR / "fund_scorecard.csv"
    if not scorecard_path.exists():
        print(f"Error: Scorecard data not found at {scorecard_path}. Please run compute_metrics.py first.")
        return None
        
    df = pd.read_csv(scorecard_path)
    
    # 1. Prepare Top 5 Funds
    top_5 = df.head(5).to_dict(orient="records")
    
    # 2. Compute Category level average metrics
    # Drop rows where critical data might be missing to ensure accurate calculations
    cat_df = df.dropna(subset=["category", "cagr_3yr_pct", "sharpe_ratio", "expense_ratio_pct"])
    cat_stats = cat_df.groupby("category").agg(
        cagr_3yr_pct=("cagr_3yr_pct", "mean"),
        sharpe_ratio=("sharpe_ratio", "mean"),
        expense_ratio_pct=("expense_ratio_pct", "mean")
    ).reset_index().to_dict(orient="records")
    
    # 3. Render HTML template using Jinja2
    template = Template(HTML_TEMPLATE)
    rendered_html = template.render(
        date_str=datetime.now().strftime("%A, %d %B %Y"),
        total_funds=len(df),
        top_funds=top_5,
        cat_stats=cat_stats
    )
    
    # Write to report file
    report_file = REPORTS_DIR / "weekly_summary.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(rendered_html)
        
    print(f"HTML Report generated and saved -> {report_file}")
    return rendered_html

def send_email(html_content, recipient_email, smtp_server, smtp_port, sender_email, sender_password):
    print(f"Attempting to send email to {recipient_email}...")
    
    # Create email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Bluestock MF Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    
    part = MIMEText(html_content, "html")
    msg.attach(part)
    
    try:
        # Establish secure connection
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print("✅ Email sent successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bluestock HTML Email Report Generator")
    parser.add_argument("--send", action="store_true", help="Send the generated HTML report via email")
    parser.add_argument("--to", type=str, help="Recipient email address")
    parser.add_argument("--server", type=str, default="smtp.gmail.com", help="SMTP server domain (default: smtp.gmail.com)")
    parser.add_argument("--port", type=int, default=587, help="SMTP server port (default: 587)")
    parser.add_argument("--user", type=str, help="SMTP sender login email")
    parser.add_argument("--password", type=str, help="SMTP sender password or App password")
    
    args = parser.parse_args()
    
    # Generate HTML report
    html_report = generate_report()
    
    if args.send:
        if not (args.to and args.user and args.password):
            print("Error: To send email, you must provide --to, --user, and --password.")
            sys.exit(1)
            
        send_email(
            html_content=html_report,
            recipient_email=args.to,
            smtp_server=args.server,
            smtp_port=args.port,
            sender_email=args.user,
            sender_password=args.password
        )
