# QSR GenAI Restaurant Assistant

[![CI](https://github.com/s73hruti/qsr-genai-restaurant-assistant/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/s73hruti/qsr-genai-restaurant-assistant/actions/workflows/ci.yml?query=branch%3Amain)

An AI-powered Quick Service Restaurant (QSR) assistant that combines Retrieval-Augmented Generation (RAG), LLM-based tool calling, machine learning models, and guardrails to support restaurant customers and managers.

An AI-powered Quick Service Restaurant (QSR) assistant that combines Retrieval-Augmented Generation (RAG), LLM-based tool calling, machine learning models, and guardrails to support restaurant customers and managers.

The project demonstrates how GenAI and traditional machine learning can work together in a restaurant operations environment.

**Live Demo:** https://qsr-genai-assistant.onrender.com/docs

---

## Project Overview

The QSR GenAI Restaurant Assistant provides a conversational interface for restaurant-related queries and operational tasks.

It combines:

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant information from a restaurant knowledge base before generating answers.
- **LLM-powered agent:** Uses tool calling to interact with knowledge retrieval and operational capabilities.
- **Machine Learning:** Supports demand forecasting, wait-time prediction, and menu recommendations.
- **Guardrails:** Applies input validation, citation checks, and output validation to reduce unsupported responses.
- **FastAPI:** Exposes the assistant and supporting capabilities through REST APIs.

The application has separate customer and manager access paths, with authentication controls for protected operations.

## Key Features

### 1. Conversational AI Assistant

- Accepts natural-language restaurant queries.
- Uses an LLM to understand questions and decide when tools are needed.
- Supports customer and manager interaction modes.
- Uses retrieved information to ground knowledge-based responses.
- Returns source citations for supported knowledge-base answers.

### 2. Retrieval-Augmented Generation (RAG)

The RAG pipeline retrieves relevant information from the restaurant knowledge base and supplies it to the LLM as context.

Retrieval capabilities include:

- Hybrid retrieval.
- TF-IDF-based text representation.
- NumPy-based vector storage.
- Query rewriting.
- Reranking of retrieved results.
- Source-aware answer generation.

The knowledge base includes information about:

- Restaurant locations and operating hours.
- Menu items and prices.
- Promotions and offers.
- Allergens and dietary information.
- Restaurant policies and procedures.

### 3. LLM Tool Calling

The conversational agent can use tools to perform specific tasks instead of relying exclusively on the language model's internal knowledge.

Available capabilities include:

| Tool | Purpose |
|---|---|
| `search_knowledge` | Retrieve restaurant information from the knowledge base |
| `forecast_demand` | Forecast restaurant demand |
| `plan_inventory` | Support inventory planning |
| `wait_time` | Estimate customer waiting time |
| `recommend` | Generate menu recommendations |
| `dynamic_menu` | Generate context-aware menu suggestions |

Tool availability depends on the user's access mode and permissions.

### 4. Machine Learning Capabilities

The project combines traditional ML models with the GenAI application.

**Demand forecasting**

Predicts restaurant demand using historical data.

Reported holdout performance:

| Metric | Result |
|---|---:|
| MAPE | 5.26% |
| Naive baseline MAPE | 9.48% |
| MAE | 18.55 |
| Holdout period | 28 days |

**Wait-time prediction**

Estimates customer waiting time.

| Metric | Result |
|---|---:|
| MAE | 0.565 minutes |
| Baseline MAE | 1.935 minutes |
| P90 coverage | 86.8% |

These are the project's reported evaluation results, not guarantees of future production performance.

**Menu recommendations**

The recommendation engine combines:

- Content-based item similarity.
- Historical popularity.
- Co-purchase patterns.
- Contextual signals such as time of day and temperature.

It supports personalized recommendations and cross-selling based on cart items.

### 5. Guardrails and Responsible Answer Generation

The assistant includes safeguards intended to improve response reliability.

These include:

- Input guardrails.
- PII redaction.
- Protection against prompt-injection attempts.
- Citation validation.
- Detection of unsupported numerical claims.
- Strict output validation.
- Fallback responses when answers cannot be sufficiently grounded.

The application is designed to prefer a safe fallback over presenting unsupported information as fact.

### 6. Authentication and Access Control

The application separates customer and manager capabilities.

Security features include:

- Manager authentication.
- User authentication for protected user-specific operations.
- Tool access restrictions based on interaction mode.
- Protected manager endpoints.

Authentication checks are retained rather than bypassed for convenience.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| API framework | FastAPI |
| GenAI | LLM integration, tool calling |
| LLM provider | Groq-compatible OpenAI API |
| LLM model | `openai/gpt-oss-20b` in the current deployment |
| RAG | Hybrid retrieval, TF-IDF, reranking, query rewriting |
| Vector storage | NumPy |
| Machine Learning | Scikit-learn, NumPy, Pandas |
| Evaluation | Local LLM evaluation, retrieval and answer-quality checks |
| Deployment | Render |
| API documentation | Swagger / OpenAPI |

The LLM backend is configurable. The current live deployment uses Groq through an OpenAI-compatible API endpoint.

---

## System Architecture

```text
                 Customer / Manager
                         |
                         v
                  FastAPI Application
                         |
                         v
                   Agent Layer
                         |
              +----------+----------+
              |                     |
              v                     v
        LLM / Tool Calling     Guardrails
              |                     |
              v                     v
        Tool Selection        Input Validation
              |
       +------+------+----------------+
       |             |                 |
       v             v                 v
 Knowledge       ML Services       Menu Tools
 Retrieval       Forecasting       Recommendations
       |          Wait-time         Dynamic Menu
       |          Inventory
       v
 Hybrid Retrieval
       |
       +-- TF-IDF
       +-- NumPy Vector Store
       +-- Query Rewriting
       +-- Reranking
       |
       v
 Retrieved Evidence
       |
       v
 LLM Response Generation
       |
       v
 Citation and Output Validation
       |
       v
 Final Response
```

---

## API and Live Deployment

The application is deployed on Render.

**API documentation:**

https://qsr-genai-assistant.onrender.com/docs

The Swagger interface allows users to explore available endpoints and test API requests.

### Example: Customer Chat

Endpoint:

`POST /chat`

Example request:

```json
{
  "message": "Is the Southpark store open 24 hours? Please explain.",
  "mode": "customer",
  "store_id": 15,
  "session_id": "demo-session-001"
}
```

The assistant can retrieve relevant store information and generate a response with supporting citations.

### Health Check

The health endpoint reports application and deployment information, including:

- LLM backend and model.
- Retrieval configuration.
- Number of indexed documents and chunks.
- Authentication configuration.
- Forecasting model metrics.
- Wait-time model metrics.

Refer to the live Swagger documentation for the currently exposed endpoint schemas and authentication requirements.

---

## Evaluation and Testing

The project includes evaluation workflows for examining the behaviour of the assistant.

Evaluation areas include:

- Retrieval quality.
- Answer correctness.
- Citation validity.
- Guardrail behaviour.
- Tool selection.
- LLM response quality.
- Latency and token usage.

Local evaluation has included Ollama-based generator and judge configurations.

The project also includes tests for:

- Agent tool routing.
- Citation repair and validation.
- Guardrail behaviour.
- Authentication.
- Machine learning components.
- API functionality.

Evaluation results should be interpreted within the scope of their test datasets and configurations. A passing test suite or successful API response does not by itself establish that every live conversational scenario is reliable.

---

## Known Limitations

The application is a portfolio and experimental project. Some capabilities require further validation.

Current limitations include:

1. **Citation formatting:** Certain LLM-generated citation formats may not be recognized by the strict citation validator, potentially causing a grounded draft to be replaced with a fallback response.
2. **Menu retrieval:** Generic menu questions may not always retrieve the most relevant menu documents.
3. **Repeated retrieval:** The agent may occasionally repeat a knowledge search instead of exploring a different query or stopping.
4. **Conversational recommendations:** Although the recommendation engine and recommendation tool exist, reliable invocation through natural-language customer chat still requires further validation.
5. **Production validation:** The live deployment has been smoke-tested, but comprehensive evaluation across a larger set of real-world conversational scenarios remains ongoing.

The output guardrails and authentication controls are intentionally retained. Unsupported responses should not be treated as verified information.

---

## Project Goals

This project explores the integration of traditional machine learning and modern GenAI systems in a practical business application.

The main engineering goals are:

- Build a retrieval-grounded conversational assistant.
- Integrate LLM tool calling with existing ML capabilities.
- Provide operational intelligence through forecasting and wait-time prediction.
- Implement source-aware answer generation.
- Add guardrails for safer and more reliable responses.
- Expose the system through REST APIs.
- Evaluate the behaviour of different LLM configurations.

---

## Future Improvements

Potential areas for further development include:

- Improve citation-format compatibility while preserving strict evidence validation.
- Improve retrieval coverage for generic menu and pricing questions.
- Improve conversational recommendation routing.
- Reduce redundant tool calls.
- Expand live evaluation with a larger and more diverse test set.
- Improve observability and production monitoring.
- Validate recommendation and personalization workflows with authenticated users.

---

## Disclaimer

This project uses a fictional restaurant dataset and synthetic business information for demonstration and experimentation. It is not affiliated with any real restaurant brand.

Model performance metrics are based on the project's evaluation datasets and should not be interpreted as guarantees of production performance.

---

## Author

**Shruti Shrivastava**

AI / ML and Generative AI Engineering Projects

