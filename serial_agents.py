# spam_langraph_example.py
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# --- 1. Define the shared state schema ---
class SpamState(TypedDict, total=False):
    input_text: str            # incoming message
    cleaned_text: str         # preprocessor output
    classifier_label: str     # "spam" / "not_spam" / "maybe"
    classifier_score: float   # optional numeric confidence 0..1
    final_label: str          # final "spam" or "not_spam"
    reason: str               # human-readable reason

# --- 2. Define node functions (synchronous functions are fine) ---
def preprocessor(state: SpamState):
    txt = state.get("input_text", "")
    # tiny cleaning: lowercase and remove excess whitespace
    cleaned = " ".join(txt.lower().split())
    # very small heuristic: remove obvious punctuation
    cleaned = cleaned.replace("\n", " ").strip()
    return {"cleaned_text": cleaned}

def classifier(state: SpamState):
    text = state.get("cleaned_text", "")
    # Simple rule-based example (replace with LLM/tool call if desired)
    spam_keywords = ["win", "free", "click", "credit", "urgent", "prize", "lottery"]
    hits = sum(1 for k in spam_keywords if k in text)
    # score between 0 and 1
    score = min(1.0, hits / 3)
    if score >= 0.6:
        label = "spam"
    elif score >= 0.3:
        label = "maybe"
    else:
        label = "not_spam"
    return {"classifier_label": label, "classifier_score": score}

def arbiter(state: SpamState):
    label = state.get("classifier_label", "not_spam")
    score = state.get("classifier_score", 0.0)
    # final decision: if classifier says "spam" or score >= 0.6 -> spam, else not_spam
    if label == "spam" or score >= 0.6:
        final = "spam"
        reason = f"classifier-> {label} (score={score:.2f})"
    elif label == "maybe":
        # tie-breaker fallback: if message contains links / many exclamation marks mark spam
        txt = state.get("cleaned_text", "")
        if ("http://" in txt or "https://" in txt) or ("!" in txt and txt.count("!") >= 2):
            final = "spam"
            reason = "maybe + suspicious punctuation/links"
        else:
            final = "not_spam"
            reason = "maybe -> treated as not_spam by fallback"
    else:
        final = "not_spam"
        reason = f"classifier-> {label} (score={score:.2f})"
    return {"final_label": final, "reason": reason}

# --- 3. Build the StateGraph ---
graph = StateGraph(SpamState)

graph.add_node("Preprocessor", preprocessor)
graph.add_node("Classifier", classifier)
graph.add_node("Arbiter", arbiter)

# connect start -> Preprocessor -> Classifier -> Arbiter -> END
graph.add_edge(START, "Preprocessor")
graph.add_edge("Preprocessor", "Classifier")
graph.add_edge("Classifier", "Arbiter")
graph.add_edge("Arbiter", END)

# compile into a runnable app
app = graph.compile()

# --- 4. Example invocation ---
if __name__ == "__main__":
    sample = {"input_text": "You WIN a FREE prize!! Click http://spam.example.com to claim!!!"}
    out = app.invoke(sample)
    print("Result state:", out)
    # expected final_label: "spam" and a reason
