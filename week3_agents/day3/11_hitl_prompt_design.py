"""
11_hitl_prompt_design.py
Demonstrates the difference between good and bad HITL prompts.
"""

# BAD HITL prompt — vague, easy to ignore
BAD_PROMPT = "The agent wants to send an email. Approve? (y/n)"

# GOOD HITL prompt — specific, contextual, actionable
GOOD_PROMPT = """
═══════════════════════════════════════
🛑 APPROVAL REQUIRED: External Email
═══════════════════════════════════════

TO: competitor-research@company.com
SUBJECT: Competitive Analysis: Tesla — Battlecard
ATTACHMENTS: tesla_battlecard_v1.pdf (124 KB)

PREVIEW:
------------------------------------------------------------------
Team,

Attached is the competitive battlecard for Tesla, covering:
- Q3 2024 revenue: $25.18B (source: Tesla 10-Q filing)
- Key strengths: Vertical integration, Supercharger network
- Key weaknesses: Production delays, pricing pressure

This analysis was generated automatically by The Agency.
------------------------------------------------------------------

⚠️ This email will be sent to 8 recipients including external consultants.
It contains competitive intelligence that may be sensitive.

✅ [1] APPROVE — Send immediately
❌ [2] REJECT — Do not send; ask agent to revise
✏️ [3] EDIT — Modify recipients or content first

Your choice: """


def format_good_hitl_prompt(action: str, args: dict, reasoning: str, preview: str) -> str:
    """Programmatically build a good HITL prompt."""
    return f"""
═══════════════════════════════════════
🛑 APPROVAL REQUIRED: {action.replace('_', ' ').title()}
═══════════════════════════════════════

ACTION DETAILS:
{json.dumps(args, indent=2)}

PREVIEW:
{preview}

⚠️ CONSEQUENCES:
{_get_consequences(action)}

✅ [1] APPROVE — Execute as proposed
❌ [2] REJECT — Cancel and ask for alternatives
✏️ [3] EDIT — Modify parameters

Your choice: """


def _get_consequences(action: str) -> str:
    """Get consequences for a specific action."""
    consequences = {
        "send_email": "This email will be sent to recipients and cannot be undone.",
        "post_slack": "This message will be visible to the entire channel.",
        "save_file": "This file will be written to the filesystem.",
        "finalize_report": "This report will be marked as final and shared."
    }
    return consequences.get(action, "This action will execute immediately if approved.")


# Example of a good HITL prompt with preview
if __name__ == "__main__":
    action = "send_email"
    args = {
        "to": "competitor-research@company.com",
        "subject": "Tesla Battlecard",
        "body": "Attached is the competitive analysis for Tesla."
    }
    reasoning = "The user asked for a battlecard to be emailed to the team."
    preview = "This email contains competitive intelligence data."
    
    print(format_good_hitl_prompt(action, args, reasoning, preview))