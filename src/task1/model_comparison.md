# Models, Tokens, and Cost - Reference Matrix

## 1. Model Families Comparison Matrix

| Model Family | Provider | Weight Architecture | Native Context Window | Core Strengths / Ideal Production Use Cases | Official Documentation Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT Family** (e.g., GPT-4o) | OpenAI | Closed-Weights (API) | 128,000 tokens | Fast response time, highly reliable for JSON data outputs, and excellent for tool connections. | [OpenAI Models Guide](https://openai.com) |
| **Claude Family** (e.g., Claude 3.5 Sonnet) | Anthropic | Closed-Weights (API) | 200,000 tokens | Excellent for writing complex code, solving difficult logic problems, and deep text analysis. | [Anthropic Claude Docs](https://anthropic.com) |
| **Gemini Family** (e.g., Gemini 1.5 Pro) | Google | Closed-Weights (API) | 1,000,000+ tokens | Massive memory size, best for analyzing long videos, audio files, and very large code repositories. | [Google AI Models](https://google.dev) |
| **Llama Family** (e.g., Llama 3.3) | Meta AI | Open-Weights (Local) | 128,000 tokens | Runs privately on local servers, completely customizable, and ensures 100% data privacy. | [Meta Llama Documentation](https://meta.com) |


## 2. Enterprise Model Selection Justification (Strict 5-Line Architecture Rule)

1. **Public Customer-Support Bot Recommendation**: GPT-4o-mini is recommended because it offers ultra-low latency, high cost-efficiency for massive traffic, and robust handling of multilingual customer queries.
2. **Justification (Support)**: Its exceptional instruction-following capability ensures polite, accurate, and consistent responses to user tickets while keeping operational costs extremely low.
3. **Private On-Premise Tool Recommendation**: Llama 3.3 (open-weights) is recommended for secure, internal corporate environments where strict data privacy and regulatory compliance are mandatory.
4. **Justification (On-Premise)**: Deploying it locally behind corporate firewalls guarantees zero data leakage of sensitive financial or personal documents to external cloud APIs.
5. **Conclusion**: This hybrid approach balances high-speed public user interaction with absolute data sovereignty for internal enterprise operations.