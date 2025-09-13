# KYC Compliance - GenAI Agent Architecture

## Design Philosophy

### Non-Functional Requirements Traceability
- **Human Process Fidelity**: Data flow must mirror human KYC officer dependency chain (identity → risk assessment → screening → investigation → documentation)
- **Chain of Custody**: Each step maps directly to human KYC officer activities, making audit trails intuitive for regulators and stakeholders
- **Process Transparency**: Stakeholders must be able to understand AI decisions because they follow familiar human workflows

### Explainability Priority
- **Natural Language Reasoning** > **Algorithmic Complexity**: A GenAI system that says "Found potential match for 'Mohammad Smith' considering common spelling variations" is infinitely more explainable than a black-box Levenshtein distance algorithm with 20 years of accumulated edge-case rules
- **Human-Readable Logic**: Prompts that mirror human thought processes ("Does this person appear on sanctions lists?") vs. cryptic algorithmic thresholds
- **Contextual Understanding**: GenAI can explain *why* it flagged something in plain language, not just *that* it was flagged

### Transformative Architecture Imperative
- **GenAI-Native Design**: Build architectures that leverage GenAI's reasoning, context understanding, and natural language capabilities from the ground up
- **Anti-Pattern**: OpenAI wrapper functions bolted onto legacy ETL pipelines (30 Lambda step functions triggered by Glue) = using smartphones only for phone calls
- **Transformative Patterns**: Ray clusters with vLLM for distributed reasoning, GraphRAG for contextual knowledge synthesis, agent-based workflows that mirror human cognitive patterns

## Preliminary Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    KYC COMPLIANCE AGENT PIPELINE                │
│                   (Human Process Fidelity)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
         ┌──────▼──────┐                ┌──────▼──────┐
         │ Identity    │                │ Risk        │
         │ Verification│◄──────────────►│ Assessment  │
         │ Agent       │                │ Agent       │
         └──────┬──────┘                └──────┬──────┘
                │                               │
         ┌──────▼──────────────────────────────▼──────┐
         │         GraphRAG Knowledge Layer           │
         │    (Sanctions + Corporate + Geo context)   │
         └──────┬──────────────────────────────┬──────┘
                │                               │
         ┌──────▼──────┐                ┌──────▼──────┐
         │ Screening   │                │Investigation│
         │ Agent       │◄──────────────►│ Agent       │
         └──────┬──────┘                └──────┬──────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                        ┌───────▼────────┐
                        │ Documentation  │
                        │ Agent          │
                        │ (Audit Trail)  │
                        └────────────────┘
```

## Component Assessment

**🟢 KEEP (Aligned with GenAI-Native Architecture):**
1. **Ray + vLLM Infrastructure** - Core distributed reasoning backbone
2. **Sanctions Database** (`watchman.db`) - 3,134+ real sanctions records for GraphRAG
3. **Pydantic Models** - Type safety for agent interfaces
4. **Test Infrastructure** - E2E pipeline validation

**🟡 TRANSFORM (Retrofit for GenAI-Native):**
1. **`src/batch_infer.py`** - Convert to KYC Agent orchestrator
2. **Data Models** - Extract for agent interface definitions

**🔴 REMOVE (ETL Anti-Patterns):**
1. **Hardcoded Data Sources** - Replace with dynamic agent queries
2. **Hash-Based Redaction** - GenAI needs contextual understanding
3. **Placeholder Logic** - Replace with reasoning agents
4. **Linear Pipeline** - Replace with agent mesh architecture


