import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing in .env file!")

client = Groq(api_key=api_key)
prompts_dir = "src/task4/prompts"
os.makedirs(prompts_dir, exist_ok=True)

print("=== Prompt Engineering Toolkit===\n")

def call_groq(prompt, temperature=0.0, top_p=1.0):
    valid_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b"
    ]
    
    for model_name in valid_models:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            continue
            
    print("   [API Error] Could not generate response.")
    return ""

print("--- Initializing Golden Dataset ---")
golden_tickets = [
    {"text": "My bill was charged twice this month.", "category": "Billing", "priority": "High"},
    {"text": "How do I update my credit card details?", "category": "Billing", "priority": "Low"},
    {"text": "I need a refund for my recent invoice.", "category": "Billing", "priority": "Medium"},
    {"text": "The payment gateway failed during checkout.", "category": "Billing", "priority": "High"},
    {"text": "Can I get an itemized statement for last year?", "category": "Billing", "priority": "Low"},
    
    {"text": "The server keeps crashing on API calls.", "category": "Technical", "priority": "High"},
    {"text": "Database latency is extremely high today.", "category": "Technical", "priority": "High"},
    {"text": "Unable to connect via SSH to production node.", "category": "Technical", "priority": "Medium"},
    {"text": "SSL certificate expired on our subdomain.", "category": "Technical", "priority": "High"},
    {"text": "Memory leak detected in microservice worker.", "category": "Technical", "priority": "Medium"},
    
    {"text": "How do I reset my account password?", "category": "Account", "priority": "Low"},
    {"text": "I want to delete my user profile permanently.", "category": "Account", "priority": "Medium"},
    {"text": "Two-factor authentication code is not arriving.", "category": "Account", "priority": "High"},
    {"text": "Can I change my registered email address?", "category": "Account", "priority": "Low"},
    {"text": "Unauthorized login attempt detected on my account.", "category": "Account", "priority": "High"}
]

resume_paragraphs = [
    "John Doe is a Senior Software Engineer with 6 years of experience in Python, Docker, and Kubernetes.",
    "Jane Smith has worked for 4 years as a Data Scientist specializing in PyTorch, Pandas, and SQL.",
    "Alex Johnson brings 8 years of DevOps expertise, proficient in Terraform, AWS, and CI/CD pipelines.",
    "Emily Davis is a Frontend Developer with 3 years of experience using React, TypeScript, and Tailwind CSS.",
    "Michael Brown has 5 years of backend engineering experience specializing in Node.js, PostgreSQL, and Redis."
]
print(f"Loaded {len(golden_tickets)} triage tickets and {len(resume_paragraphs)} candidate resumes.\n")

print("--- Evaluating 3 Prompt Versions on Golden Set ---")

prompt_v1 = "Classify the following ticket into one of these categories: Billing, Technical, Account.\nTicket: {text}"
prompt_v2 = "Role: Expert Customer Support Triage AI.\nTask: Classify the given support ticket.\nConstraints: Output ONLY the exact category name from [Billing, Technical, Account] with no extra text.\nTicket: {text}"
prompt_v3 = """You are an expert customer support classifier. Categorize the ticket into Billing, Technical, or Account.

Example 1:
Ticket: "My credit card was charged twice for subscription."
Category: Billing

Example 2:
Ticket: "API endpoints are returning 500 internal server errors."
Category: Technical

Example 3:
Ticket: "I cannot log into my profile because my password expired."
Category: Account

Ticket: {text}
Category:"""

with open(f"{prompts_dir}/prompt_v1_basic.txt", "w", encoding="utf-8") as f:
    f.write(prompt_v1)
with open(f"{prompts_dir}/prompt_v2_structured.txt", "w", encoding="utf-8") as f:
    f.write(prompt_v2)
with open(f"{prompts_dir}/prompt_v3_fewshot.txt", "w", encoding="utf-8") as f:
    f.write(prompt_v3)

def evaluate_prompt(template):
    correct = 0
    for ticket in golden_tickets:
        p = template.format(text=ticket["text"])
        res = call_groq(p, temperature=0.0)
        if ticket["category"].lower() in res.lower():
            correct += 1
    return correct

score_v1 = evaluate_prompt(prompt_v1)
score_v2 = evaluate_prompt(prompt_v2)
score_v3 = evaluate_prompt(prompt_v3)

print(f"v1 (Basic): {score_v1}/15 ({int(score_v1/15*100)}%)")
print(f"v2 (Structured): {score_v2}/15 ({int(score_v2/15*100)}%)")
print(f"v3 (Few-Shot): {score_v3}/15 ({int(score_v3/15*100)}%)\n")

print("--- Chain-of-Thought (CoT) Testing ---")
word_problems = [
    "A store has 35 books. They sell 12 in the morning and receive 20 more in the afternoon. How many books do they have now?",
    "John has 4 boxes of apples, each containing 6 apples. He gives 5 apples to his friend. How many apples does he have left?",
    "A train travels 60 miles in 2 hours. If it maintains the same speed, how long will it take to travel 180 miles?",
    "Sarah has $50. She spends $14 on a book and $22 on groceries. How much money does she have remaining?",
    "There are 3 rows of chairs with 8 chairs in each row. If 5 chairs are broken and removed, how many usable chairs are left?"
]

for idx, wp in enumerate(word_problems, 1):
    direct_ans = call_groq(wp, temperature=0.0)
    cot_ans = call_groq(f"Think step by step to solve this problem: {wp}", temperature=0.0)
    print(f"Problem {idx}:")
    print(f"  Direct: {direct_ans}")
    print(f"  CoT:    {cot_ans}")
print()

print("--- Sampling & Creativity Evaluation ---")
creative_prompt = "Write ONLY a single creative sci-fi tagline (one sentence max) about quantum AI taking over data centers. Respond STRICTLY in English with no explanation."

print("Testing Temperature Variations (0.0, 0.7, 1.0):")
for temp in [0.0, 0.7, 1.0]:
    res = call_groq(creative_prompt, temperature=temp)
    print(f"Temp {temp}: {res}")

print("\nComparing Top-P (0.3 vs 0.95):")
print(f"Top-P 0.3 (Deterministic): {call_groq(creative_prompt, temperature=0.7, top_p=0.3)}")
print(f"Top-P 0.95 (Diverse):      {call_groq(creative_prompt, temperature=0.7, top_p=0.95)}\n")

print("--- Pydantic Schema Validation & Bad Prompt Mitigation ---")

class ResumeSchema(BaseModel):
    name: str = Field(description="Full name of candidate")
    skills: list[str] = Field(description="Technical skills list")
    years_experience: int = Field(description="Years of professional experience")

plain_invalid = 0
for resume in resume_paragraphs:
    p = f"Extract candidate info: {resume}"
    raw = call_groq(p, temperature=0.0)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        ResumeSchema.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError):
        plain_invalid += 1

json_mode_invalid = 0
for idx, resume in enumerate(resume_paragraphs, 1):
    p = f"Extract candidate details from this text and output STRICT VALID JSON only matching keys name (string), skills (array of strings), years_experience (integer): {resume}"
    raw = call_groq(p, temperature=0.0)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        validated = ResumeSchema.model_validate_json(raw)
        print(f"-> Resume {idx} Validated: name='{validated.name}' skills={validated.skills} years_experience={validated.years_experience}")
    except (ValidationError, json.JSONDecodeError):
        json_mode_invalid += 1

print(f"\nPlain: {plain_invalid}/5 invalid JSON | JSON mode: {json_mode_invalid}/5 invalid")
print(f"Total Invalid Outputs Count: {json_mode_invalid}/5\n")

bad_prompts_fixes = {
    "Bad Prompt 1": "Extract data.",
    "Fixed Prompt 1": "Extract candidate info strictly as a JSON object matching schema {name, skills[], years_experience}.",
    "Bad Prompt 2": "Tell me about the resume.",
    "Fixed Prompt 2": "Role: HR Parser. Analyze the text and output valid structured JSON only.",
    "Bad Prompt 3": "Give me skills.",
    "Fixed Prompt 3": "Extract all technical skills into a JSON array without conversational filler."
}

print("Rewritten Bad Prompts & Improvements:")
for k, v in bad_prompts_fixes.items():
    print(f"-> {k} -> {v}")

print("\n Complete assessment execution finished successfully.")