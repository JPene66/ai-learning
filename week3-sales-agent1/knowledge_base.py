"""
knowledge_base.py
Apex's internal "company knowledge" - simulated data standing in for what
would, in production, be a CRM export, a docs site, and a real graph DB.
Everything here is free, local, and deterministic so the whole project
runs without any paid API beyond OpenAI's LLM calls.

Three pieces:
  CASE_STUDIES / PRICING_FAQ  - retrieved during objection handling (Close stage)
  PAST_DEALS                   - retrieved during qualification for self-querying comparison
  ACCOUNT_GRAPH                 - a lightweight knowledge graph for multi-hop
                                   relational queries (Week 3, Skill 2, Concept 6: GraphRAG)
"""

CASE_STUDIES = [
    {"id": "cs_01", "industry": "fintech", "pain_point": "manual compliance reporting",
     "text": "A mid-market fintech (200 employees) cut compliance report prep time from 6 hours to 40 minutes per week after adopting our platform, freeing their risk team to focus on actual risk analysis instead of formatting spreadsheets."},
    {"id": "cs_02", "industry": "healthtech", "pain_point": "slow patient data reconciliation",
     "text": "A healthtech provider serving 40 clinics reduced patient record reconciliation errors by 68% within the first quarter, avoiding two near-miss compliance incidents their team flagged as previously routine."},
    {"id": "cs_03", "industry": "logistics", "pain_point": "manual dispatch scheduling",
     "text": "A regional logistics company with 15 depots automated dispatch scheduling and reduced late deliveries by 22% in the first two months, while cutting 3 hours of daily manual coordination work."},
    {"id": "cs_04", "industry": "fintech", "pain_point": "customer onboarding delays",
     "text": "A digital lender shortened average customer onboarding from 4 days to under 6 hours by automating document verification, directly increasing loan conversion rate."},
    {"id": "cs_05", "industry": "retail", "pain_point": "inventory forecasting accuracy",
     "text": "A multi-location retailer improved inventory forecast accuracy by 31%, reducing both stockouts and overstock costs across its 60-store network within one selling season."},
]

PRICING_FAQ = [
    {"id": "faq_01", "topic": "discount_policy",
     "answer": "Standard discounts up to 15% are available for annual commitments. Larger discounts require a business justification and manager approval."},
    {"id": "faq_02", "topic": "contract_length",
     "answer": "We offer monthly, annual, and 2-year terms. Annual and multi-year terms include the best per-seat pricing."},
    {"id": "faq_03", "topic": "competitor_comparison",
     "answer": "Compared to legacy providers, our platform typically reduces implementation time from 6-8 weeks to under 2 weeks, with no long-term lock-in beyond the contract term chosen."},
    {"id": "faq_04", "topic": "security_compliance",
     "answer": "We are SOC 2 Type II compliant with annual third-party audits, and support SSO, audit logging, and role-based access control on all paid tiers."},
    {"id": "faq_05", "topic": "cancellation_policy",
     "answer": "Annual contracts can be cancelled with 30 days notice before renewal; no cancellation fee applies if notice is given in that window."},
]

# Past deals - used for the self-querying retriever exercise (Concept 4) and as
# seed data for long-term memory (Concept 3). "outcome" and "objection_handled"
# are exactly the kind of episodic detail that makes future objection-handling smarter.
PAST_DEALS = [
    {"id": "deal_01", "industry": "fintech", "deal_size_usd": 42000, "outcome": "won",
     "objection_handled": "pricing", "notes": "Closed after offering annual term with 10% discount; case study cs_01 was decisive."},
    {"id": "deal_02", "industry": "fintech", "deal_size_usd": 15000, "outcome": "lost",
     "objection_handled": "security_compliance", "notes": "Lost to a competitor with a longer compliance track record; SOC 2 answer came too late in the cycle."},
    {"id": "deal_03", "industry": "healthtech", "deal_size_usd": 61000, "outcome": "won",
     "objection_handled": "contract_length", "notes": "Prospect wanted monthly; closed on annual after showing per-seat savings math directly."},
    {"id": "deal_04", "industry": "logistics", "deal_size_usd": 28000, "outcome": "won",
     "objection_handled": "competitor_comparison", "notes": "Competitor's implementation timeline was the deciding factor once compared side by side."},
    {"id": "deal_05", "industry": "retail", "deal_size_usd": 19500, "outcome": "lost",
     "objection_handled": "pricing", "notes": "Budget was cut mid-cycle; no amount of discounting would have closed this one - a timing issue, not a pricing issue."},
    {"id": "deal_06", "industry": "fintech", "deal_size_usd": 55000, "outcome": "won",
     "objection_handled": "contract_length", "notes": "2-year term closed with a 12% discount after finance team compared TCO to their legacy vendor."},
]

# --- Lightweight knowledge graph: entities + relationships (Concept 6: GraphRAG) ---
# Real GraphRAG uses Neo4j/Neptune; this in-memory adjacency structure demonstrates
# the same core idea - multi-hop traversal - without requiring a graph database.
ACCOUNT_GRAPH = {
    "nodes": {
        "acme_fintech": {"type": "company", "industry": "fintech"},
        "acme_capital_partners": {"type": "company", "industry": "fintech", "note": "parent company of acme_fintech"},
        "jane_doe": {"type": "contact", "role": "VP Finance", "company": "acme_fintech"},
        "john_smith": {"type": "contact", "role": "CTO", "company": "acme_fintech"},
        "rival_finco": {"type": "company", "industry": "fintech", "note": "competitor to acme_fintech"},
        "healthbridge": {"type": "company", "industry": "healthtech"},
        "priya_anand": {"type": "contact", "role": "Director of Ops", "company": "healthbridge"},
        "logi_move": {"type": "company", "industry": "logistics"},
        "logi_move_west": {"type": "company", "industry": "logistics", "note": "subsidiary of logi_move"},
    },
    "edges": [
        ("jane_doe", "works_at", "acme_fintech"),
        ("john_smith", "works_at", "acme_fintech"),
        ("acme_fintech", "subsidiary_of", "acme_capital_partners"),
        ("acme_fintech", "competitor_of", "rival_finco"),
        ("priya_anand", "works_at", "healthbridge"),
        ("logi_move_west", "subsidiary_of", "logi_move"),
    ],
}


def find_related(entity_id: str, max_hops: int = 2):
    """
    Multi-hop traversal over ACCOUNT_GRAPH - e.g. find_related('jane_doe')
    surfaces her company, that company's parent, and its competitor, even
    though none of those are directly connected to her in one hop. This is
    the core GraphRAG idea: some questions need relationship-following, not
    just similarity search.
    """
    visited = {entity_id}
    frontier = {entity_id}
    related = []

    for _ in range(max_hops):
        next_frontier = set()
        for src, rel, dst in ACCOUNT_GRAPH["edges"]:
            if src in frontier and dst not in visited:
                related.append({"from": src, "relation": rel, "to": dst})
                next_frontier.add(dst)
                visited.add(dst)
            if dst in frontier and src not in visited:
                related.append({"from": dst, "relation": f"<-{rel}-", "to": src})
                next_frontier.add(src)
                visited.add(src)
        frontier = next_frontier
        if not frontier:
            break

    return related