import re

def detect_ai_generated_text(text):
    # Simple keyword and structure-based heuristics
    generic_phrases = ["in conclusion", "as mentioned earlier", "to summarize", "generally speaking"]
    unnatural_patterns = re.compile(r"(repeating pattern|unnatural sentence structure)")
    
    # Check for generic phrases
    for phrase in generic_phrases:
        if phrase in text.lower():
            return "Possible AI-generated content detected based on generic phrasing."
    
    # Check for unnatural patterns
    if unnatural_patterns.search(text):
        return "Possible AI-generated content detected based on unnatural patterns."
    
    return "Content likely human-written or no indicators found."

# Example usage
text = "In conclusion, as mentioned earlier, AI-generated text is usually generic."
print(detect_ai_generated_text(text))
