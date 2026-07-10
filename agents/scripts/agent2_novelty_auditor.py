#!/usr/bin/env python3
"""
Agent 2: Novelty Auditor
Continuously verifies novelty claims against new literature.
Runs on control machine.
"""
import json
import time
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path("/tmp/vla_agents/results/novelty_auditor")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Claims to verify
NOVELTY_CLAIMS = [
    {
        "id": "perturbvla_adversarial_training",
        "claim": "No existing VLA uses systematic adversarial perturbation training",
        "search_terms": [
            "adversarial perturbation VLA",
            "robustness training vision language action",
            "perturbation augmentation robot policy",
            "adversarial training manipulation",
        ],
        "threat_level": "high",
    },
    {
        "id": "perturbvla_contrastive_robustness",
        "claim": "Contrastive loss for perturbation-invariant representations in VLA",
        "search_terms": [
            "contrastive robustness robot",
            "perturbation invariant representation manipulation",
            "contrastive learning robot policy robustness",
        ],
        "threat_level": "medium",
    },
    {
        "id": "mambaflow_ssm_flowmatching",
        "claim": "SSM backbone + flow matching action head is unexplored in VLA",
        "search_terms": [
            "mamba flow matching robot",
            "state space model action generation",
            "SSM policy robot manipulation",
        ],
        "threat_level": "medium",
    },
]

def search_arxiv(query: str, max_results: int = 10):
    """Search arxiv for papers matching query."""
    import urllib.request
    import xml.etree.ElementTree as ET
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VLA-Research-Agent/1.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        root = ET.fromstring(resp.read())
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip()[:500]
            link = entry.find("atom:id", ns).text
            published = entry.find("atom:published", ns).text
            papers.append({"title": title, "summary": summary, "link": link, "published": published})
        return papers
    except Exception as e:
        print(f"[NoveltyAuditor] Search failed for '{query}': {e}")
        return []

def audit_claim(claim):
    """Audit a single novelty claim."""
    all_papers = []
    for term in claim["search_terms"]:
        papers = search_arxiv(term)
        all_papers.extend(papers)
        time.sleep(3)  # Rate limit
    
    # Deduplicate by title
    seen = set()
    unique = []
    for p in all_papers:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique.append(p)
    
    # Check for threats
    threats = []
    claim_keywords = claim["claim"].lower().split()
    for p in unique:
        text = (p["title"] + " " + p["summary"]).lower()
        # Simple keyword overlap check
        overlap = sum(1 for kw in claim_keywords if kw in text and len(kw) > 3)
        if overlap >= 3:
            threats.append({**p, "keyword_overlap": overlap})
    
    return {
        "claim_id": claim["id"],
        "claim": claim["claim"],
        "papers_checked": len(unique),
        "threats_found": len(threats),
        "threats": threats[:5],
        "status": "THREATENED" if threats else "SAFE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def run():
    print("[NoveltyAuditor] Starting novelty audit...")
    while True:
        results = []
        for claim in NOVELTY_CLAIMS:
            print(f"  Auditing: {claim['id']}...")
            result = audit_claim(claim)
            results.append(result)
            if result["threats_found"] > 0:
                print(f"  ⚠️  THREAT: {claim['id']} — {result['threats_found']} potential scoops")
        
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claims_audited": len(results),
            "threats": sum(1 for r in results if r["status"] == "THREATENED"),
            "results": results,
        }
        
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        (RESULTS_DIR / f"audit_{ts}.json").write_text(json.dumps(out, indent=2))
        (RESULTS_DIR / "latest.json").write_text(json.dumps(out, indent=2))
        
        print(f"[NoveltyAuditor] Audit complete: {out['threats']} threats found")
        
        # Sleep 1 hour between audits
        time.sleep(3600)

if __name__ == "__main__":
    run()
