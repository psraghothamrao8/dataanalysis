import sys
import os
# Add path to allow imports
sys.path.append(os.getcwd())

from app.services.agent_service import analyze_situation_and_decide, query_llama3

def verify_llm():
    print("--- Verifying Agent / LLM Service ---")
    
    # 1. Test Raw Connection
    print("1. Testing Ollama Connection...")
    response = query_llama3("Are you working? Reply with 'Yes'.")
    print(f"   Raw Response: {response[:50]}...")
    
    if response == "Error":
        print("   [WARN] Ollama is not reachable or model missing. Service will use FALLBACK mode.")
    else:
        print("   [PASS] Ollama is responding.")

    # 2. Test Full Logic
    print("\n2. Testing Logic (analyze_situation_and_decide)...")
    try:
        decision = analyze_situation_and_decide()
        print(f"   Decision: {decision.get('recommended_action')}")
        print(f"   Issues Found: {len(decision.get('issues_list', []))}")
        
        # Check if it was a fallback decision or LLM decision
        # We can infer fallback if functionality works despite Error above
        print("   [PASS] Agent logic executed successfully.")
    except Exception as e:
        print(f"   [FAIL] Agent logic crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_llm()
