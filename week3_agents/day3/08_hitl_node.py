"""
08_hitl_node.py
Implements the HITL checkpoint node that pauses for human approval.
"""

import json
from langgraph.types import interrupt
from langchain_core.messages import AIMessage

from 05_state_schema import AgentState
from 01_task_models import Task


def hitl_checkpoint_node(state: AgentState):
    """
    Pause execution and request human approval for high-stakes actions.
    """
    # Get the pending action from the last AI message
    ai_messages = [m for m in state["messages"] if m.type == "ai"]
    if not ai_messages:
        return {"messages": []}
    
    last_ai_msg = ai_messages[-1]
    
    # Extract tool calls that need approval
    pending_actions = []
    if hasattr(last_ai_msg, "tool_calls") and last_ai_msg.tool_calls:
        high_stakes_tools = {"send_email", "post_slack", "save_file", "finalize_report"}
        for tc in last_ai_msg.tool_calls:
            if tc.get("name") in high_stakes_tools:
                pending_actions.append({
                    "tool": tc["name"],
                    "args": tc.get("args", {}),
                    "reasoning": last_ai_msg.content[:200]  # First 200 chars
                })
    
    if not pending_actions:
        # No high-stakes actions — pass through
        return {"messages": []}
    
    # Build the approval prompt
    action = pending_actions[0]  # Handle one at a time for simplicity
    
    approval_prompt = f"""
═══════════════════════════════════════
🛑 HUMAN APPROVAL REQUIRED
═══════════════════════════════════════

📋 ACTION: {action['tool']}
📦 PARAMETERS: {json.dumps(action['args'], indent=2)}

🧠 REASONING:
{action['reasoning']}

⚠️ CONSEQUENCES:
- This action will execute immediately if approved
- It may send data externally, modify files, or finalize outputs
- Rejection will return control to the agent to replan

✅ OPTIONS:
[1] APPROVE — Execute the action as proposed
[2] REJECT — Cancel the action and let the agent replan
[3] EDIT — Modify the parameters before executing

Enter your choice (1/2/3): """
    
    # INTERRUPT: This pauses the graph and returns the prompt to the user
    human_response = interrupt({
        "prompt": approval_prompt,
        "action": action,
        "state_snapshot": state
    })
    
    # Execution resumes here after the human responds
    choice = human_response.get("choice")
    
    if choice == "1":
        # APPROVE — return empty, let the tool node execute
        print("✅ Human approved. Proceeding...")
        return {"messages": []}
    
    elif choice == "2":
        # REJECT — inject a rejection message into the conversation
        rejection_msg = f"The user rejected the {action['tool']} action. Please find an alternative approach."
        return {"messages": [AIMessage(content=rejection_msg)]}
    
    elif choice == "3":
        # EDIT — use the modified parameters
        edited_args = human_response.get("edited_args", action["args"])
        print(f"✏️ Human edited parameters: {edited_args}")
        # In production, you'd modify the state to replace the tool call args
        return {"messages": []}
    
    else:
        return {"messages": []}