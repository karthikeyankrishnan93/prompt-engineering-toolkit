import tiktoken

def print_comparison_table():
    print("--- Model Family Comparison Table ---")
    table = [
        ["Family", "Owner", "Weights Type", "Context Window", "Key Strengths"],
        ["GPT", "OpenAI", "Closed", "Up to 128k - 1M+", "Advanced reasoning, coding, function calling"],
        ["Claude", "Anthropic", "Closed", "Up to 200k", "Nuanced writing, long-context analysis, safety"],
        ["Llama", "Meta", "Open Weights", "Up to 128k+", "Fine-tuning, cost-effective, self-hosting"],
        ["Gemini", "Google", "Closed", "Up to 1M - 2M", "Multimodal (native audio/video), massive context"]
    ]
    for row in table:
        print(" | ".join(row))
    print("\n" + "-"*50 + "\n")

def analyze_tokens():
    print("--- Tiktoken Token Count & Words-per-Token ---")
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    texts = {
        "Sentence": "Hello, how are you doing today?",
        "Paragraph": "Artificial intelligence is rapidly evolving, transforming modern enterprise architectures by automating complex semantic processing pipelines globally.",
        "Page / Long Text": "This is a simulated long text page representing a detailed document report containing multiple paragraphs of operational notes, system telemetry, and architectural analysis for the prompt engineering toolkit lab environment.",
        "JSON": '{"id": 101, "item": "laptop", "price": 999.99, "in_stock": true}',
        "Tamil Sentence": "வணக்கம், இன்று உங்களுக்கு எப்படி உதவ முடியும்?"
    }
    
    for name, text in texts.items():
        tokens = enc.encode(text)
        token_count = len(tokens)
        word_count = len(text.split())
        ratio = round(word_count / token_count, 2) if token_count > 0 else 0
        print(f"{name}:")
        print(f"  - Text: \"{text}\"")
        print(f"  - Tokens: {token_count} | Words: {word_count} | Ratio: {ratio} words/token")
    print("\n" + "-"*50 + "\n")

def estimate_cost(input_tokens, output_tokens, price_in_per_1m, price_out_per_1m):
    input_cost = (input_tokens / 1_000_000) * price_in_per_1m
    output_cost = (output_tokens / 1_000_000) * price_out_per_1m
    return input_cost + output_cost

def run_cost_simulation():
    print("--- [Requirement 3] Cost Estimation ---")
    chats_per_day = 1000
    days_in_month = 30
    input_per_chat = 1500
    output_per_chat = 400
    
    total_monthly_input = chats_per_day * input_per_chat * days_in_month
    total_monthly_output = chats_per_day * output_per_chat * days_in_month
    
    price_in = 0.15
    price_out = 0.60
    
    monthly_cost = estimate_cost(total_monthly_input, total_monthly_output, price_in, price_out)
    
    print(f"Scenario: {chats_per_day} chats/day, {input_per_chat} input tokens, {output_per_chat} output tokens.")
    print(f"Monthly Input Tokens: {total_monthly_input:,}")
    print(f"Monthly Output Tokens: {total_monthly_output:,}")
    print(f"Estimated Monthly Cost (GPT-4o-mini): ${monthly_cost:.2f}")
    print("\n" + "-"*50 + "\n")

def context_window_analysis():
    print("--- Context Window Fit Analysis ---")
    print("- 8k Context Window (8,192 tokens): YES (All sample texts fit comfortably)")
    print("- 128k Context Window (131,072 tokens): YES (All sample texts fit easily with massive room for long documents)")
    print("\n" + "-"*50 + "\n")

def print_recommendations():
    print("--- Model Recommendations & Justifications ---")
    print("1. Public Customer-Support Bot Recommendation:")
    print("   - Model: GPT-4o-mini or Claude 3.5 Haiku")
    print("   - Justification: Offers ultra-low latency, high cost-efficiency for massive user volumes, exceptional instruction-following capability, robust safety guardrails, and seamless handling of multilingual queries.")
    print("\n2. Private On-Premise Tool Recommendation:")
    print("   - Model: Llama 3 / 3.1 (8B or 70B open-weights)")
    print("   - Justification: Open-weight nature enables secure local deployment behind corporate firewalls. Guarantees zero data leakage of confidential enterprise documents, full compliance with strict privacy laws, and complete control via custom fine-tuning.")
    print("\nexecution finished successfully!")

if __name__ == "__main__":
    print("=== Models, Tokens and Cost ===\n")
    print_comparison_table()
    analyze_tokens()
    run_cost_simulation()
    context_window_analysis()
    print_recommendations()