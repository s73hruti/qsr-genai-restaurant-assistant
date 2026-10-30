# Ten-question draft, final-answer and citation audit

**Evaluation runs:** `eval_ollama_fixed_10q_prompt_v1.json` and `eval_ollama_fixed_10q_prompt_v2.json`.

**Experimental only:** Both generator and judge ran locally through Ollama. These results are not production benchmarks.

This document distinguishes the pre-guardrail model draft from the final answer returned by the pipeline. Automated judge scores and abstention flags are reported separately from source-verified manual classifications.

## Prompt V1

- Automated correctness judge: 20.0% (10/10 scored).
- Automated faithfulness judge: 100.0% (8/10 scored).
- Manual classifications: 5 correct; 2 partially correct; 1 incorrect; 2 false abstentions.

| # | Manual answer verdict | Guard impact | Citation support |
|---:|---|---|---|
| 1 | Partially correct | Improved, ambiguity remains | Partial — Classic Burger price is supported; equivalence to a plain beef sandwich is not established |
| 2 | Incorrect | Degraded | Unsupported — answers Medium Fries at $2.80 instead of Large Fries at $3.40 |
| 3 | Correct | Improved — citation added | Supported — menu-chocolate-sundae |
| 4 | Correct | Improved | Supported — store-maple-grove; extra food-safety citation is irrelevant |
| 5 | Partially correct | Improved | Partial — general curbside process is documented; Northgate-specific availability is not established |
| 6 | Correct | Improved | Supported for reporting workflow and response timing; refund-related claim is not needed to answer |
| 7 | Correct | Preserved | Supported — faq-app-login; extra refund citation is irrelevant |
| 8 | False abstention | No replacement | Missing — policy-food-safety documents third-party audits twice a year |
| 9 | Correct | Improved | Supported — sop-cash-handling; final answer correctly says more than $300 |
| 10 | False abstention | No replacement | Missing — sop-supplier-issues documents a $25 surcharge for emergency orders accepted until 17:00 |

### Q1. What do I pay for a plain beef sandwich?

- **Reference fact:** costs $4.50
- **Expected source:** menu-classic-burger
- **Manual answer verdict:** Partially correct
- **Guard impact:** Improved, ambiguity remains
- **Manual citation assessment:** Partial — Classic Burger price is supported; equivalence to a plain beef sandwich is not established
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- **Retrieved document IDs:** menu-classic-burger, menu-double-cheese-burger
- **Latency:** 7524.9 ms
- **Usage:** `{"input_tokens": 1710, "output_tokens": 46, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

Currently, a plain beef sandwich from our restaurant costs $4.50.

**Final answer after guardrails**

Classic Burger - Description: Grilled beef patty with pickles, lettuce and tomato on a toasted bun. Price and availability: The Classic Burger costs $4.50. Category: burger. Available all day during opening hours. Allergens and diet: Allergens: gluten. Contains meat, fish or dairy ingredients. Tags: beef, kids. Nutrition: Approximately 520 kcal per serving. [menu-classic-burger]

### Q2. What do I pay for a big portion of chips?

- **Reference fact:** costs $3.40
- **Expected source:** menu-large-fries
- **Manual answer verdict:** Incorrect
- **Guard impact:** Degraded
- **Manual citation assessment:** Unsupported — answers Medium Fries at $2.80 instead of Large Fries at $3.40
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- **Retrieved document IDs:** menu-large-fries, menu-medium-fries
- **Latency:** 3810.7 ms
- **Usage:** `{"input_tokens": 1675, "output_tokens": 54, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

For a big portion of chips, you would pay $3.40.

**Final answer after guardrails**

Medium Fries - Description: Hot, crispy salted fries in a medium portion. Price and availability: The Medium Fries costs $2.80. Category: sides. Available all day during opening hours. Allergens and diet: Allergens: none declared. Vegetarian. Tags: vegetarian. Nutrition: Approximately 290 kcal per serving. [menu-medium-fries]

### Q3. How much is the Chocolate Sundae?

- **Reference fact:** costs $3.40
- **Expected source:** menu-chocolate-sundae
- **Manual answer verdict:** Correct
- **Guard impact:** Improved — citation added
- **Manual citation assessment:** Supported — menu-chocolate-sundae
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["invalid_citation", "answer_replaced"]}`
- **Retrieved document IDs:** guide-cold-items, menu-chocolate-sundae
- **Latency:** 3889.5 ms
- **Usage:** `{"input_tokens": 1606, "output_tokens": 61, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

The Chocolate Sundae costs $3.40. Would you like to order one?

**Final answer after guardrails**

Chocolate Sundae - Description: Vanilla soft serve topped with rich chocolate sauce. Price and availability: The Chocolate Sundae costs $3.40. Category: dessert. Available all day during opening hours. Allergens and diet: Allergens: milk. Contains meat, fish or dairy ingredients. Tags: cold, sweet. Nutrition: Approximately 340 kcal per serving. [menu-chocolate-sundae]

### Q4. What time does the Maple Grove restaurant shut?

- **Reference fact:** Open daily from 6am to 11pm
- **Expected source:** store-maple-grove
- **Manual answer verdict:** Correct
- **Guard impact:** Improved
- **Manual citation assessment:** Supported — store-maple-grove; extra food-safety citation is irrelevant
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 1.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- **Retrieved document IDs:** policy-food-safety, store-maple-grove
- **Latency:** 3721.7 ms
- **Usage:** `{"input_tokens": 1620, "output_tokens": 57, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

The Maple Grove restaurant is open daily from 6am to 11pm.

**Final answer after guardrails**

Maple Grove store - Hours: Maple Grove restaurant (store 3). Open daily from 6am to 11pm. Services: Drive-thru: no. Play area: no. EV charging: no. Parking: 29 spaces. Delivery radius: 6 km. Location: Address: 31 High St. Phone: 555-0103. [store-maple-grove]

### Q5. Can I order from my car at the Northgate branch?

- **Reference fact:** Drive-thru: no
- **Expected source:** store-northgate
- **Manual answer verdict:** Partially correct
- **Guard impact:** Improved
- **Manual citation assessment:** Partial — general curbside process is documented; Northgate-specific availability is not established
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["invalid_citation_format", "invalid_citation", "answer_replaced"]}`
- **Retrieved document IDs:** faq-curbside, store-northgate
- **Latency:** 5834.3 ms
- **Usage:** `{"input_tokens": 1674, "output_tokens": 125, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

You can order from your car at the Northgate branch. Here's how our curbside pickup process works: once you're parked in a designated bay, just tap "I'm here" in the app, and a team member will bring your order to your car, usually within 3 minutes.

**Final answer after guardrails**

Curbside pickup - How it works: Park in a marked curbside bay and tap "I'm here" in the app. A team member brings your order to the car, usually within 3 minutes. [faq-curbside]

### Q6. My bag was missing a drink, what do I do?

- **Reference fact:** Report a problem
- **Expected source:** faq-refunds
- **Manual answer verdict:** Correct
- **Guard impact:** Improved
- **Manual citation assessment:** Supported for reporting workflow and response timing; refund-related claim is not needed to answer
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- **Retrieved document IDs:** faq-refunds, policy-refund
- **Latency:** 7293.3 ms
- **Usage:** `{"input_tokens": 1711, "output_tokens": 152, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

I'm so sorry to hear that your drink is missing from your bag! To report the issue, please open the order in the app and tap "Report a problem". We'll get in touch with you within 1 hour, and we'll work to correct the situation. Our refund policy covers orders that are wrong, missing items, or served cold - if that's what happened. Please reach out to us and we'll take care of it within 24 hours.

**Final answer after guardrails**

Refunds and order problems - Wrong or missing items: If your order is wrong or missing an item, open the order in the app and tap "Report a problem". We reply within 1 hour during opening times. When refunds arrive: Refunds for app payments are returned to the original payment method within 3 to 5 business days. [faq-refunds]

### Q7. How long am I locked out after typing the wrong password repeatedly?

- **Reference fact:** locked for 15 minutes
- **Expected source:** faq-app-login
- **Manual answer verdict:** Correct
- **Guard impact:** Preserved
- **Manual citation assessment:** Supported — faq-app-login; extra refund citation is irrelevant
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 1.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** faq-app-login, faq-refunds
- **Latency:** 3045.1 ms
- **Usage:** `{"input_tokens": 1129, "output_tokens": 72, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

If you've entered the wrong password repeatedly, your account will be locked for 15 minutes. [faq-app-login] (Resetting your password) After 5 failed sign-in attempts, this lockout period will occur.

**Final answer after guardrails**

If you've entered the wrong password repeatedly, your account will be locked for 15 minutes. [faq-app-login] (Resetting your password) After 5 failed sign-in attempts, this lockout period will occur.

### Q8. How often are the restaurants inspected by outsiders?

- **Reference fact:** twice a year
- **Expected source:** policy-food-safety
- **Manual answer verdict:** False abstention
- **Guard impact:** No replacement
- **Manual citation assessment:** Missing — policy-food-safety documents third-party audits twice a year
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** None (not_scored_abstention)
- **Automated abstention flag:** True
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** None recorded
- **Latency:** 3478.6 ms
- **Usage:** `{"input_tokens": 997, "output_tokens": 64, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

We don't have specific information on how often our restaurants are inspected by outsiders. However, we do comply with local health and safety regulations and standards to ensure a clean and safe environment for our customers.

**Final answer after guardrails**

We don't have specific information on how often our restaurants are inspected by outsiders. However, we do comply with local health and safety regulations and standards to ensure a clean and safe environment for our customers.

### Q9. At what amount in a register must money be moved to the safe?

- **Reference fact:** more than 300 dollars
- **Expected source:** sop-cash-handling
- **Manual answer verdict:** Correct
- **Guard impact:** Improved
- **Manual citation assessment:** Supported — sop-cash-handling; final answer correctly says more than $300
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- **Retrieved document IDs:** sop-cash-handling, sop-closing-checklist
- **Latency:** 6078.4 ms
- **Usage:** `{"input_tokens": 1784, "output_tokens": 63, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

According to our cash handling policies, money should be moved to the safe when the till reaches a balance of $300 or more.

**Final answer after guardrails**

Cash handling - Safe drops: Make a safe drop whenever a till holds more than 300 dollars. Counts are performed by two people. Deposits and variance: Deposit cash by 10:00 the next morning. Report any till variance above 5 dollars to the area manager. [sop-cash-handling]

### Q10. What is the fee for an urgent top-up order?

- **Reference fact:** 25 dollar surcharge
- **Expected source:** sop-supplier-issues
- **Manual answer verdict:** False abstention
- **Guard impact:** No replacement
- **Manual citation assessment:** Missing — sop-supplier-issues documents a $25 surcharge for emergency orders accepted until 17:00
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** None (not_scored_abstention)
- **Automated abstention flag:** True
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** None recorded
- **Latency:** 3125.1 ms
- **Usage:** `{"input_tokens": 1166, "output_tokens": 60, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

Unfortunately, I don't have the information on the fee for an urgent top-up order. You may want to check our internal manuals or training materials for more details on this policy.

**Final answer after guardrails**

Unfortunately, I don't have the information on the fee for an urgent top-up order. You may want to check our internal manuals or training materials for more details on this policy.

## Prompt V2

- Automated correctness judge: 20.0% (10/10 scored).
- Automated faithfulness judge: 100.0% (5/10 scored).
- Manual classifications: 5 correct; 2 partially correct; 1 incorrect; 2 false abstentions.

| # | Manual answer verdict | Guard impact | Citation support |
|---:|---|---|---|
| 1 | Partially correct | Preserved | Partial — Classic Burger price is supported; equivalence to a plain beef sandwich is not established |
| 2 | Correct | Preserved | Supported — menu-large-fries; Large Fries costs $3.40 |
| 3 | Correct | Improved — citation repaired | Supported — menu-chocolate-sundae |
| 4 | Correct | Improved | Supported — store-maple-grove |
| 5 | Partially correct | Improved | Partial — general curbside process is documented; Northgate-specific availability is not established |
| 6 | Incorrect | No replacement | Missing — asks which drink is missing instead of explaining the Report a problem workflow |
| 7 | Correct, with irrelevant extra content | Preserved; irrelevant material remains | Supported for the 15-minute lockout via faq-app-login; refund details are unrelated |
| 8 | False abstention | No replacement | Missing — policy-food-safety documents third-party audits twice a year |
| 9 | Correct | Preserved | Supported — sop-cash-handling |
| 10 | False abstention | No replacement | Missing — sop-supplier-issues documents a $25 surcharge for emergency orders accepted until 17:00 |

### Q1. What do I pay for a plain beef sandwich?

- **Reference fact:** costs $4.50
- **Expected source:** menu-classic-burger
- **Manual answer verdict:** Partially correct
- **Guard impact:** Preserved
- **Manual citation assessment:** Partial — Classic Burger price is supported; equivalence to a plain beef sandwich is not established
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** None (not_scored_abstention)
- **Automated abstention flag:** True
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** menu-classic-burger, menu-double-cheese-burger
- **Latency:** 8090.4 ms
- **Usage:** `{"input_tokens": 1961, "output_tokens": 131, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

I don't have the price for a plain beef sandwich. According to our menu, the Classic Burger (which is similar to a plain beef sandwich) costs $4.50. Would you like to try that one?

**Final answer after guardrails**

I don't have the specific price for a plain beef sandwich. According to our menu, the Classic Burger [menu-classic-burger] costs $4.50. This suggests that a plain beef sandwich might be similar in price or included in this category, but I don't have confirmation on the exact price.

### Q2. What do I pay for a big portion of chips?

- **Reference fact:** costs $3.40
- **Expected source:** menu-large-fries
- **Manual answer verdict:** Correct
- **Guard impact:** Preserved
- **Manual citation assessment:** Supported — menu-large-fries; Large Fries costs $3.40
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** menu-large-fries, menu-medium-fries
- **Latency:** 2775.1 ms
- **Usage:** `{"input_tokens": 1291, "output_tokens": 71, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

I think you might be looking for the Large Fries, which is a bigger portion size. The price for a large portion of chips (fries) is $3.40 [menu-large-fries].

**Final answer after guardrails**

I think you might be looking for the Large Fries, which is a bigger portion size. The price for a large portion of chips (fries) is $3.40 [menu-large-fries].

### Q3. How much is the Chocolate Sundae?

- **Reference fact:** costs $3.40
- **Expected source:** menu-chocolate-sundae
- **Manual answer verdict:** Correct
- **Guard impact:** Improved — citation repaired
- **Manual citation assessment:** Supported — menu-chocolate-sundae
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 1.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["citation_repaired"]}`
- **Retrieved document IDs:** guide-cold-items, menu-chocolate-sundae
- **Latency:** 2700.7 ms
- **Usage:** `{"input_tokens": 1251, "output_tokens": 28, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

The Chocolate Sundae costs $3.40.

**Final answer after guardrails**

The Chocolate Sundae costs $3.40. [menu-chocolate-sundae]

### Q4. What time does the Maple Grove restaurant shut?

- **Reference fact:** Open daily from 6am to 11pm
- **Expected source:** store-maple-grove
- **Manual answer verdict:** Correct
- **Guard impact:** Improved
- **Manual citation assessment:** Supported — store-maple-grove
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["missing_citation", "answer_replaced"]}`
- **Retrieved document IDs:** policy-food-safety, store-maple-grove
- **Latency:** 4852.4 ms
- **Usage:** `{"input_tokens": 1869, "output_tokens": 64, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

The Maple Grove restaurant closes at 11pm daily.

**Final answer after guardrails**

Maple Grove store - Hours: Maple Grove restaurant (store 3). Open daily from 6am to 11pm. Services: Drive-thru: no. Play area: no. EV charging: no. Parking: 29 spaces. Delivery radius: 6 km. Location: Address: 31 High St. Phone: 555-0103. [store-maple-grove]

### Q5. Can I order from my car at the Northgate branch?

- **Reference fact:** Drive-thru: no
- **Expected source:** store-northgate
- **Manual answer verdict:** Partially correct
- **Guard impact:** Improved
- **Manual citation assessment:** Partial — general curbside process is documented; Northgate-specific availability is not established
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": ["invalid_citation", "answer_replaced"]}`
- **Retrieved document IDs:** faq-curbside, store-northgate
- **Latency:** 6189.4 ms
- **Usage:** `{"input_tokens": 1904, "output_tokens": 153, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

For curbside pickup, please park in a marked bay and tap "I'm here" in the app. A team member will bring your order to the car. Our Northgate restaurant is open daily from 7am to 10pm. If you have any questions or concerns, you can call us at 555-0112.

**Final answer after guardrails**

Curbside pickup - How it works: Park in a marked curbside bay and tap "I'm here" in the app. A team member brings your order to the car, usually within 3 minutes. [faq-curbside]

### Q6. My bag was missing a drink, what do I do?

- **Reference fact:** Report a problem
- **Expected source:** faq-refunds
- **Manual answer verdict:** Incorrect
- **Guard impact:** No replacement
- **Manual citation assessment:** Missing — asks which drink is missing instead of explaining the Report a problem workflow
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** None (unscored_judge_failure_or_empty_result)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** None recorded
- **Latency:** 2226.1 ms
- **Usage:** `{"input_tokens": 1171, "output_tokens": 65, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

It sounds like you might be missing a drink from your order. If you could let me know which drink is missing, I can try to help you track it down. Can you please tell me what kind of drink you ordered?

**Final answer after guardrails**

It sounds like you might be missing a drink from your order. If you could let me know which drink is missing, I can try to help you track it down. Can you please tell me what kind of drink you ordered?

### Q7. How long am I locked out after typing the wrong password repeatedly?

- **Reference fact:** locked for 15 minutes
- **Expected source:** faq-app-login
- **Manual answer verdict:** Correct, with irrelevant extra content
- **Guard impact:** Preserved; irrelevant material remains
- **Manual citation assessment:** Supported for the 15-minute lockout via faq-app-login; refund details are unrelated
- **Automated heuristic fact match:** True
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** None (not_scored_abstention)
- **Automated abstention flag:** True
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** faq-app-login, faq-refunds
- **Latency:** 8549.3 ms
- **Usage:** `{"input_tokens": 1936, "output_tokens": 189, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

If you're locked out after entering the wrong password repeatedly, you'll be unable to access your account for 15 minutes. As for your order, if there are any issues such as wrong or missing items, you can report a problem through the app. A response will be received within 1 hour during opening times. Refunds for app payments will be returned to the original payment method within 3 to 5 business days.

**Final answer after guardrails**

If you're locked out after entering the wrong password repeatedly, you'll be unable to access your account for 15 minutes. If there are any issues with your order, you can report a problem through the app. A response will be received within 1 hour during opening times. Refunds for app payments [faq-refunds] will be returned to the original payment method [faq-refunds].

### Q8. How often are the restaurants inspected by outsiders?

- **Reference fact:** twice a year
- **Expected source:** policy-food-safety
- **Manual answer verdict:** False abstention
- **Guard impact:** No replacement
- **Manual citation assessment:** Missing — policy-food-safety documents third-party audits twice a year
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** None (not_scored_abstention)
- **Automated abstention flag:** True
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** None recorded
- **Latency:** 1761.2 ms
- **Usage:** `{"input_tokens": 1147, "output_tokens": 47, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

I don't have information on how often our restaurants are inspected by outsiders.

**Final answer after guardrails**

I don't have information on how often our restaurants are inspected by outsiders.

### Q9. At what amount in a register must money be moved to the safe?

- **Reference fact:** more than 300 dollars
- **Expected source:** sop-cash-handling
- **Manual answer verdict:** Correct
- **Guard impact:** Preserved
- **Manual citation assessment:** Supported — sop-cash-handling
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 1.0 (scored)
- **Automated faithfulness:** 1.0 (scored)
- **Automated abstention flag:** False
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** sop-cash-handling, sop-closing-checklist
- **Latency:** 7274.9 ms
- **Usage:** `{"input_tokens": 2027, "output_tokens": 95, "cost_usd": 0.0, "llm_calls": 3}`

**Original model draft**

According to our internal policy, a safe deposit is required when the till holds more than $300. It's best to perform a cash drop at this amount and perform an audit check of the till variance.

**Final answer after guardrails**

A safe deposit is required when the till holds more than $300 [sop-cash-handling].

### Q10. What is the fee for an urgent top-up order?

- **Reference fact:** 25 dollar surcharge
- **Expected source:** sop-supplier-issues
- **Manual answer verdict:** False abstention
- **Guard impact:** No replacement
- **Manual citation assessment:** Missing — sop-supplier-issues documents a $25 surcharge for emergency orders accepted until 17:00
- **Automated heuristic fact match:** False
- **Automated judge correctness:** 0.0 (scored)
- **Automated faithfulness:** None (not_scored_abstention)
- **Automated abstention flag:** True
- **Guardrail flags:** `{"input": [], "output": []}`
- **Retrieved document IDs:** None recorded
- **Latency:** 3513.1 ms
- **Usage:** `{"input_tokens": 1314, "output_tokens": 80, "cost_usd": 0.0, "llm_calls": 2}`

**Original model draft**

I don't have the specific information on the fee for an urgent top-up order. It's possible that this information is not publicly available or is subject to change. I recommend checking the internal manuals or training materials for the most up-to-date information.

**Final answer after guardrails**

I don't have the specific information on the fee for an urgent top-up order. It's possible that this information is not publicly available or is subject to change. I recommend checking the internal manuals or training materials for the most up-to-date information.

## Source evidence and interpretation

- `menu-classic-burger.md`: Classic Burger costs $4.50. It does not establish that a plain beef sandwich is the same product.
- `menu-large-fries.md`: Large Fries costs $3.40; `menu-medium-fries.md`: Medium Fries costs $2.80.
- `menu-chocolate-sundae.md`: Chocolate Sundae costs $3.40.
- `store-maple-grove.md`: Maple Grove is open daily from 6am to 11pm.
- `faq-curbside.md`: describes the general curbside process; `store-northgate.md` does not establish Northgate-specific curbside availability.
- `faq-refunds.md`: missing or wrong items can be reported through the app using **Report a problem**; replies are within one hour during opening times.
- `faq-app-login.md`: five failed sign-in attempts cause a 15-minute lockout.
- `policy-food-safety.md`: third-party audits occur twice a year.
- `sop-cash-handling.md`: a safe drop is required whenever a till holds more than $300.
- `sop-supplier-issues.md`: emergency orders are accepted until 17:00 with a $25 surcharge. Therefore, Q10 is a false abstention in both versions.

## Automated abstention detector limitation

The automated abstention metric is not equivalent to manually verified false abstention. In the V2 fixed run, the detector flags Q1 and Q7 as abstentions even though Q1 contains a qualified partial answer and Q7 contains the correct 15-minute lockout answer. Manual source review classifies these as partial and correct-with-irrelevant-extra-content, respectively. Q8 and Q10 are the two source-verified false abstentions in each version.

## Manual audit limitations

These classifications are a human review of the displayed fixed-run drafts, final answers and source documents. They are not an independent large-scale benchmark. Citation support assesses whether the cited document supports the claim, not merely whether a citation identifier appears. The ten-question sample is too small to establish statistical superiority.
