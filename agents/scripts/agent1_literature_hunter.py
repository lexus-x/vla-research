#!/usr/bin/env python3
"""
Agent 1: Literature Hunter
Continuously scans for new VLA papers, cross-domain ideas, and novelty threats.
Runs on control machine (no GPU needed).
"""
import json
import time
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path("/tmp/vla_agents/results/literature_hunter")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def scan_arxiv():
    """Scan arxiv for new VLA-related papers."""
    import urllib.request
    import xml.etree.ElementTree as ET
    
    queries = [
        "ti:vla+AND+cat:cs.RO",
        "ti:vision+AND+ti:language+AND+ti:action+AND+cat:cs.RO",
        "ti:embodied+AND+ti:foundation+AND+cat:cs.RO",
        "ti:robot+AND+ti:policy+AND+cat:cs.RO",
        "ti:flow+AND+ti:matching+AND+ti:robot",
    ]
    
    papers = []
    for q in queries:
        url = f"http://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate&sortOrder=descending&max_results=20"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VLA-Research-Agent/1.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            root = ET.fromstring(resp.read())
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip()[:300]
                published = entry.find("atom:published", ns).text
                link = entry.find("atom:id", ns).text
                papers.append({
                    "title": title,
                    "summary": summary,
                    "published": published,
                    "link": link,
                    "query": q,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        except Exception as e:
            print(f"[LiteratureHunter] Query failed: {q}: {e}")
    return papers

def check_novelty_threats(papers):
    """Flag papers that threaten PerturbVLA novelty."""
    threat_keywords = [
        "adversarial perturbation", "robustness training", "perturbation robust",
        "augmentation training", "contrastive robustness", "adversarial training robot"
    ]
    threats = []
    for p in papers:
        text = (p["title"] + " " + p["summary"]).lower()
        for kw in threat_keywords:
            if kw in text:
                threats.append({**p, "threat_keyword": kw})
                break
    return threats

def run():
    print("[LiteratureHunter] Starting literature scan...")
    while True:
        papers = scan_arxiv()
        threats = check_novelty_threats(papers)
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_papers": len(papers),
            "threats": threats,
            "papers": papers[:50],  # Keep top 50
        }
        
        out_file = RESULTS_DIR / f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        out_file.write_text(json.dumps(result, indent=2))
        
        # Summary
        summary = {
            "last_scan": datetime.now(timezone.utc).isoformat(),
            "papers_found": len(papers),
            "threats_found": len(threats),
            "threat_titles": [t["title"] for t in threats[:5]],
        }
        (RESULTS_DIR / "latest.json").write_text(json.dumps(summary, indent=2))
        
        print(f"[LiteratureHunter] Scanned {len(papers)} papers, {len(threats)} threats")
        
        # Sleep 30 min between scans
        time.sleep(1800)

if __name__ == "__main__":
    run()
