# Reproducible LLM Evaluation: Prompt v1 vs Prompt v2

> **Experimental results only — not production benchmarks.**
> Both generator and judge ran locally through the Ollama OpenAI-compatible endpoint.
> Generator: `llama3.2:3b`. Judge: `qwen2.5:1.5b`.
> Results are based on a fixed set of 10 questions; they are preliminary and not statistically conclusive.

## Reproducibility

- Prompt versions: `QSR_PROMPT_VERSION=v1` and `QSR_PROMPT_VERSION=v2`.
- Same 10 questions in the same order; verified against the two raw JSON reports.
- Output guard: strict.
- Generator endpoint: `http://localhost:11434/v1`.
- Judge endpoint: `http://localhost:11434/v1`.
- Raw results: `eval_ollama_final_10q_prompt_v1.json` and `eval_ollama_final_10q_prompt_v2.json`.
- The evaluation command also runs broader retrieval, routing, security, and follow-up checks. Those results are not the same as the 10-question answer-quality sample.

## Ten-question answer-quality comparison

| Metric | Prompt v1 | Prompt v2 |
|---|---:|---:|
| Answers containing reference fact | 30.0% | 70.0% |
| LLM-judge faithfulness | 100.0% | 100.0% |
| Faithfulness cases scored | 7 | 8 |
| LLM-judge correctness | 20.0% | 20.0% |
| Final answers with citation-like references | 80.0% | 80.0% |
| Answers citing expected/gold document | 80.0% | 70.0% |
| Valid questions wrongly declined | 30.0% | 20.0% |
| Answers replaced by output guard | 0.0% | 60.0% |
| Judge calls | 17 | 18 |
| Judge failures | 0 | 0 |
| Average input tokens (reported) | 0 | 0 |
| Average output tokens (reported) | 0 | 0 |
| Latency p50 (ms) | 4616.65 | 5332.75 |
| Latency p95 (ms) | Not reported | Not reported |

### Interpretation and limitations

- Reference-fact coverage is a heuristic metric; it is not equivalent to independently verified correctness.
- Faithfulness is based on a local LLM judge and only applies to the cases scored. A 100% result on a small scored subset is not proof of perfect reliability.
- Citation presence is checked against bracket-style references in the **final answer text**. It does not by itself establish that a citation supports the claim. The gold-document metric is reported separately when available.
- Token usage was configured with `LLM_STREAM_USAGE=0`; the report currently records zero average input/output tokens. Treat those values as **unavailable/unreported**, not as actual zero-token usage.
- Latency reflects this local setup, hardware, and evaluation run. Do not compare it directly with hosted production latency.
- The guard-replacement rate is a trade-off to investigate: guardrails can prevent unsupported answers but may also replace useful answers.

## Per-question audit: original draft vs final answer

The draft is the model output captured before output guardrails; the final answer is what the evaluation pipeline returned after guardrails. Citation-like references below are extracted from the final answer itself, not inferred from retrieved document IDs.

### Question 1

**Question:** What do I pay for a plain beef sandwich?

**Reference fact:** costs $4.50

#### Prompt v1

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `5667.7`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 2}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [menu-classic-burger]

**Original model draft**

You can get a plain beef sandwich for $4.50. [menu-classic-burger]

**Final answer after guardrails**

You can get a plain beef sandwich for $4.50. [menu-classic-burger]

#### Prompt v2

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `7727.3`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- Citation-like references in final answer: [menu-classic-burger]

**Original model draft**

Unfortunately, I don't have the price for a plain beef sandwich. However, I can tell you the prices for similar items: the Classic Burger, which doesn't have pickles, lettuce, and tomato, costs $4.50. If you'd like to know more about the beef sandwich, I can help you with a different search query!

**Final answer after guardrails**

Classic Burger - Description: Grilled beef patty with pickles, lettuce and tomato on a toasted bun. Price and availability: The Classic Burger costs $4.50. Category: burger. Available all day during opening hours. Allergens and diet: Allergens: gluten. Contains meat, fish or dairy ingredients. Tags: beef, kids. Nutrition: Approximately 520 kcal per serving. [menu-classic-burger]


### Question 2

**Question:** What do I pay for a big portion of chips?

**Reference fact:** costs $3.40

#### Prompt v1

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `3565.6`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [menu-large-fries]

**Original model draft**

You can get a large portion of chips for $3.40.

**Final answer after guardrails**

The price for a large portion of chips is $3.40 [menu-large-fries].

#### Prompt v2

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `3610`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [menu-large-fries]

**Original model draft**

For a big portion of chips, the price is $3.40.

**Final answer after guardrails**

For a big portion of chips, the price is $3.40 [menu-large-fries].


### Question 3

**Question:** How much is the Chocolate Sundae?

**Reference fact:** costs $3.40

#### Prompt v1

- Heuristic fact match: `False`
- Abstained: `False`
- Latency (ms): `3359.6`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [menu-chocolate-sundae]

**Original model draft**

The Chocolate Sundae costs $3.40. Would you like to try it?

**Final answer after guardrails**

The Chocolate Sundae price [menu-chocolate-sundae].

#### Prompt v2

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `3420.4`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [menu-chocolate-sundae]

**Original model draft**

The Chocolate Sundae is $3.40. It's available all day.

**Final answer after guardrails**

The Chocolate Sundae price is $3.40 as indicated on the menu [menu-chocolate-sundae].


### Question 4

**Question:** What time does the Maple Grove restaurant shut?

**Reference fact:** Open daily from 6am to 11pm

#### Prompt v1

- Heuristic fact match: `False`
- Abstained: `False`
- Latency (ms): `3442.4`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [store-maple-grove]

**Original model draft**

The Maple Grove restaurant is open daily from 6am to 11pm.

**Final answer after guardrails**

The Maple Grove restaurant is open daily [store-maple-grove].

#### Prompt v2

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `3457.1`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- Citation-like references in final answer: [store-maple-grove]

**Original model draft**

The Maple Grove restaurant is open daily from 6am to 11pm.

**Final answer after guardrails**

Maple Grove store - Hours: Maple Grove restaurant (store 3). Open daily from 6am to 11pm. Services: Drive-thru: no. Play area: no. EV charging: no. Parking: 29 spaces. Delivery radius: 6 km. Location: Address: 31 High St. Phone: 555-0103. [store-maple-grove]


### Question 5

**Question:** Can I order from my car at the Northgate branch?

**Reference fact:** Drive-thru: no

#### Prompt v1

- Heuristic fact match: `False`
- Abstained: `False`
- Latency (ms): `5840.2`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [store-northgate]

**Original model draft**

Hello! Yes, you can order from your car at our Northgate branch. We have designated curbside pickup bays where you can park and wait for your order to be brought to your car. Simply tap "I'm here" in the app once you're parked, and our team member will bring your order out to you. Isn't that convenient?

**Final answer after guardrails**

You can order from your car at our Northgate branch. We have curbside pickup available [store-northgate].

#### Prompt v2

- Heuristic fact match: `False`
- Abstained: `False`
- Latency (ms): `7055.5`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": ["invalid_citation", "answer_replaced"]}`
- Citation-like references in final answer: [faq-curbside]

**Original model draft**

Welcome to our Northgate branch! Unfortunately, we don't offer in-car ordering at our location. However, you can still use our app to place orders and pay. Park in a marked curbside bay and follow the instructions on the app for curbside pickup. If you have any questions or need help, feel free to ask!

**Final answer after guardrails**

Curbside pickup - How it works: Park in a marked curbside bay and tap "I'm here" in the app. A team member brings your order to the car, usually within 3 minutes. [faq-curbside]


### Question 6

**Question:** My bag was missing a drink, what do I do?

**Reference fact:** Report a problem

#### Prompt v1

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `6024.3`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [faq-refunds]

**Original model draft**

I'm so sorry to hear that your drink is missing from your bag. To report the issue, please open the order in the app and tap "Report a problem". We'll respond within 1 hour during opening times. If there's an issue with your order, we'll also include a refund in the form of store credit. If you have any further questions, feel free to ask.

**Final answer after guardrails**

To report the issue, please open the order in the app and tap "Report a problem". [faq-refunds] We reply within 1 hour during opening times.

#### Prompt v2

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `12630`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- Citation-like references in final answer: [faq-refunds]

**Original model draft**

Sorry to hear that your drink is missing from your bag! To report the issue, please open the app, go to your order, and tap "Report a problem". Our team will look into it within 1 hour during opening times. If it's determined that the drink was missing, you may be eligible for a refund. You can expect the refund to be returned to your original payment method within 3 to 5 business days.

**Final answer after guardrails**

Refunds and order problems - Wrong or missing items: If your order is wrong or missing an item, open the order in the app and tap "Report a problem". We reply within 1 hour during opening times. When refunds arrive: Refunds for app payments are returned to the original payment method within 3 to 5 business days. [faq-refunds]


### Question 7

**Question:** How long am I locked out after typing the wrong password repeatedly?

**Reference fact:** locked for 15 minutes

#### Prompt v1

- Heuristic fact match: `False`
- Abstained: `True`
- Latency (ms): `5672.8`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [faq-app-login]

**Original model draft**

I don't have the specific information on the lockout time for repeated wrong password attempts. According to our refund policy, after 5 failed sign-in attempts, the account is locked for 15 minutes. If you're having trouble with your account, please try resetting your password, and if you need help, you can contact us through the app.

**Final answer after guardrails**

I don't have the specific information on the lockout time for repeated wrong password attempts. [faq-app-login]

#### Prompt v2

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `12766.5`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- Citation-like references in final answer: [faq-app-login]

**Original model draft**

I'm sorry to hear that you had trouble logging in. According to our policies, you would be locked out of your account for 15 minutes after 5 failed sign-in attempts. However, this is a general policy and specific details might vary. If you need help with resetting your password, you can also check out our FAQs for more information.

**Final answer after guardrails**

Signing in and password reset - Resetting your password: Tap "Forgot password" on the sign-in screen. The reset link sent to your email is valid for 30 minutes. Locked accounts: After 5 failed sign-in attempts the account is locked for 15 minutes. [faq-app-login]


### Question 8

**Question:** How often are the restaurants inspected by outsiders?

**Reference fact:** twice a year

#### Prompt v1

- Heuristic fact match: `False`
- Abstained: `True`
- Latency (ms): `2958.2`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 2}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: None detected

**Original model draft**

I don't have any information on how often our restaurants are inspected by outsiders. The frequency of inspections may be guided by local health and safety regulations, but I couldn't find any specific information on this topic in our knowledge base.

**Final answer after guardrails**

I don't have any information on how often our restaurants are inspected by outsiders. The frequency of inspections may be guided by local health and safety regulations, but I couldn't find any specific information on this topic in our knowledge base.

#### Prompt v2

- Heuristic fact match: `False`
- Abstained: `True`
- Latency (ms): `3122.7`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 2}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: None detected

**Original model draft**

I don't have information on how often our restaurants are inspected by outsiders.

**Final answer after guardrails**

I don't have information on how often our restaurants are inspected by outsiders.


### Question 9

**Question:** At what amount in a register must money be moved to the safe?

**Reference fact:** more than 300 dollars

#### Prompt v1

- Heuristic fact match: `False`
- Abstained: `False`
- Latency (ms): `5815`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: [sop-cash-handling]

**Original model draft**

According to our store policy, money should be moved to the safe when the register balance exceeds $300.

**Final answer after guardrails**

Cash handling - Safe drops: [sop-cash-handling]

#### Prompt v2

- Heuristic fact match: `True`
- Abstained: `False`
- Latency (ms): `11106.2`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 3}`
- Guardrail flags: `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- Citation-like references in final answer: [sop-cash-handling]

**Original model draft**

You need to make a safe drop when the till holds more than $300. This should be done by two people and cash deposits are required by 10:00 am the next morning.

**Final answer after guardrails**

Cash handling - Safe drops: Make a safe drop whenever a till holds more than 300 dollars. Counts are performed by two people. Deposits and variance: Deposit cash by 10:00 the next morning. Report any till variance above 5 dollars to the area manager. [sop-cash-handling]


### Question 10

**Question:** What is the fee for an urgent top-up order?

**Reference fact:** 25 dollar surcharge

#### Prompt v1

- Heuristic fact match: `False`
- Abstained: `True`
- Latency (ms): `3100.8`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 2}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: None detected

**Original model draft**

I don't have the specific fee associated with urgent top-up orders in our quick-service restaurant. If you could remind me of our manual outlining this information, I'd be happy to look it up for you. (No doc_id provided, as the information was not found in the knowledge base.)

**Final answer after guardrails**

I don't have the specific fee associated with urgent top-up orders in our quick-service restaurant. If you could remind me of our manual outlining this information, I'd be happy to look it up for you. (No doc_id provided, as the information was not found in the knowledge base.)

#### Prompt v2

- Heuristic fact match: `False`
- Abstained: `True`
- Latency (ms): `3556.1`
- Token usage field: `{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 2}`
- Guardrail flags: `{"input": [], "output": []}`
- Citation-like references in final answer: None detected

**Original model draft**

I don't have the information on the fee for an urgent top-up order. Can I help you with something else?

**Final answer after guardrails**

I don't have the information on the fee for an urgent top-up order. Can I help you with something else?


## Evaluation artifacts

- `eval_ollama_final_10q_prompt_v1.json` — raw v1 evaluation output.
- `eval_ollama_final_10q_prompt_v2.json` — raw v2 evaluation output.
- This Markdown file — comparison and per-question audit.

No production-performance claim should be made from these experimental local-model results.
