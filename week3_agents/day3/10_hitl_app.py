"""
10_hitl_app.py
Handles interrupts in the application layer (CLI, API, UI).
"""

import json
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from 09_hitl_graph import build_hitl_graph


def run_with_hitl(thread_id: str, user_message: str):
    """
    Run the agent with HITL handling.
    This is how you'd implement it in a CLI, Streamlit app, or API.
    """
    graph = build_hitl_graph()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {"messages": [HumanMessage(content=user_message)]}
    
    # Use stream to handle interrupts gracefully
    for event in graph.stream(inputs, config=config, stream_mode="values"):
        print(f"Event: {event.get('messages', [])[-1].content[:100] if event.get('messages') else 'No messages'}")
        
        # Check if this event contains an interrupt
        # (In production, you'd check the event for interrupt data)
        if "interrupt" in str(event).lower():
            print("🛑 INTERRUPT DETECTED")
            
            # In a real implementation, you'd display the prompt and collect input
            # For this example, we'll simulate human approval
            choice = input("Your choice (1-3): ").strip()
            
            if choice == "3":
                edited = input("Enter edited JSON args: ").strip()
                command = {"choice": "3", "edited_args": json.loads(edited)}
            else:
                command = {"choice": choice}
            
            # Resume the graph with the human's command
            for resume_event in graph.stream(
                Command(resume=command),
                config=config,
                stream_mode="values"
            ):
                print(f"Resumed: {resume_event.get('messages', [])[-1].content[:100] if resume_event.get('messages') else 'No messages'}")


def run_with_hitl_stream(thread_id: str, user_message: str):
    """
    Alternative implementation using the streaming API.
    """
    graph = build_hitl_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {"messages": [HumanMessage(content=user_message)]}
    
    # Use astream for more control
    for event in graph.astream(inputs, config=config):
        for node_name, node_output in event.items():
            print(f"Node: {node_name}")
            
            # Check for interrupt
            if "__interrupt__" in node_output:
                interrupt_data = node_output["__interrupt__"][0].value
                print(f"🛑 INTERRUPT: {interrupt_data.get('prompt', 'No prompt')[:100]}...")
                
                # Simulate human response
                choice = input("Your choice: ").strip()
                if choice == "3":
                    edited = input("Edited args: ").strip()
                    command = {"choice": "3", "edited_args": json.loads(edited)}
                else:
                    command = {"choice": choice}
                
                # Resume
                for resume_event in graph.astream(Command(resume=command), config=config):
                    print(f"Resume event: {resume_event}")


if __name__ == "__main__":
    # Test with a simple message
    run_with_hitl("session_001", "Research Tesla and email the battlecard to the team")