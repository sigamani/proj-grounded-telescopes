# Philosophy

This repository is not intended to be another generic Ray + vLLM demonstration. Its purpose is to show that a complete inference pipeline — encompassing orchestration, serving, and monitoring — can be constructed in days rather than months, provided teams use standard open-source components intelligently and focus their engineering effort on genuine differentiators. The guiding principle is that small RegTech startups must prioritize efficiency and explainability over scale. Cost, agility, and auditability are not secondary concerns in this space; they are the core requirements.

The design choices are deliberate. Hyperscale cloud providers such as AWS and GCP have become prohibitively expensive and operationally cumbersome for GenAI workloads, often charging 5–10× more for the same hardware while imposing delays measured in weeks. GPU marketplaces such as Vast.ai, RunPod, and Nebius provide equivalent compute capacity at a fraction of the cost with near-instant provisioning. Kubernetes deployment makes the stack portable across hyperscalers, on-premise, or bare metal. Crucially, it avoids entanglement with proprietary services such as Vertex AI or SageMaker, which add recurring costs without improving institutional knowledge or unique capability.

The core architectural stance is equally clear: smaller models (8B–14B parameters) and offline batch processing are not a limitation, but a discipline. If a model cannot be reliably evaluated and improved at this scale with 500–1000 substantive real-world cases per week, then moving to a 70B model is unjustifiable — both economically and operationally. Bigger models do not guarantee better outcomes unless the system is already delivering verifiable accuracy, alignment, and explainability at smaller scales. The principal risks to this project are not in the infrastructure, which is commodity and open-source, but in three areas: the quality of the models themselves, the suitability of the accuracy metrics used, and access to representative training and evaluation data. Unless these are addressed systematically, the system cannot improve meaningfully, regardless of hardware or orchestration.

This repository therefore represents more than technical scaffolding. It is a legal-style proposition: that GenAI infrastructure can and should remain open, efficient, and strategically defensible. That means resisting vendor lock-in, avoiding unexamined escalation to larger models, and building processes where monitoring, feedback, and auditability are the foundation for every scaling decision.

However, higher reward comes with higher risk, data provenance and legal admissibility: sanitising inputs and using Pydantic schemas is good engineering discipline, but if inputs include third-party documents (shareholder lists, registry extracts, etc.), we must be certain of their legal status. If we store and process them without a clear chain of licensing or consent, we could run afoul of data protection law (GDPR, DPA 2018) or contractual restrictions. Cloud-agnostic deployments are a strength, but they also make it harder to demonstrate uniform compliance across vendors.

Second, audit trail sufficiency: the commitment to “every decision must be logged and traceable” is right, but regulators will not care only about logs; they will want those logs to be immutable, tamper-proof, and tied to cryptographic guarantees. Without this, audit trails can be challenged as incomplete or manipulated. There is a risk here that “traceable” in practice does not meet the evidentiary standard required in enforcement actions.

Third, explainability threshold: the idea that a jury of non-experts would reach the same conclusion is compelling, but difficult to demonstrate. If your verdict system (ACCEPT | REJECT | REVIEW) is challenged, you may need to prove that the model’s explanation is not only understandable but also consistent with statutory obligations — e.g. AMLD5/6 standards or FCA guidance. That is a higher bar than “does an intern agree?” Regulators will expect deterministic reasoning for certain risk flags (e.g. PEP status, sanctions matches) and may not accept purely probabilistic explanations.

Fourth, scaling philosophy: restricting to 8B–14B models is pragmatic, but it raises a litigation risk if, say, a REJECT verdict is later overturned and the counterparty claims that the decision was made using an “underpowered” system. Even if technically unjustified, such arguments can gain traction. You need to show that the choice of model size is a deliberate, documented tradeoff — not a cost-saving shortcut.

Fifth, monitoring and metrics: Prometheus/Grafana are excellent for engineers, but regulators and auditors are not engineers. If the system cannot translate those metrics into human-readable compliance dashboards or exportable evidence packs, you will be left with “logs nobody can read.” That undermines the very transparency we claim is central.

Finally, open-source reliance: this is a double-edged sword. It makes us independent of vendor lock-in, but it also means we assume liability if something goes wrong. If a model update introduces bias, or if an OSS dependency has a security flaw, we cannot simply point to AWS or OpenAI. Responsibility rests with us. This is not a reason to abandon the approach, but it does need to be acknowledged — and mitigated through governance and controls.

⸻

Non-Functional Requirements

Three principles govern system quality: traceability, consistency, and explainability.
	•	Traceability: Every action must be logged and reconstructable. Inputs, transformations, model versions, risk signals, and final verdicts must all be captured with timestamps, ensuring a complete audit trail. No decision may leave the chain of custody unexplained.
	•	Consistency: Identical inputs must yield identical outputs. Rules for data processing and risk scoring must be standardized, version-controlled, and synchronized across all sub-components. No subsystem may act on stale or divergent reference data.
	•	Explainability: Every decision must be accompanied by clear, human-readable reasoning and supporting evidence. Deterministic string-matching rules alone are insufficient; they are not inherently explainable. By contrast, LLM-driven explanations — provided they are grounded and auditable — can often exceed human analysts in consistency while remaining comprehensible to non-experts.

These requirements extend to operational practices. Containers must be versioned, serialized, and published (e.g., to Docker Hub) so any job can be re-run and verified, including the precise Python, PyTorch, and CUDA versions used. The system must remain open-source in design, ensuring the ability to reconstitute the entire stack without reliance on a vendor that may deprecate models or APIs at will.

Finally, the scaling philosophy is intentional. The unique selling proposition lies in fine-tuning and validating models in the 8B–14B range, not in pursuing vanity deployments of 400B-parameter models. Real-world workloads are modest but meaningful — on the order of 500–1000 jobs per week — and should form the baseline for iteration. Scaling should only occur once evidence from these cohorts demonstrates clear, measurable gains. Monitoring is non-negotiable: Prometheus, Grafana, and OpenTelemetry must expose every referenced metric so that any stakeholder can interrogate system performance without ambiguity.

⸻

Functional Requirements

The system begins with the intake of structured and unstructured inputs. All incoming data must be sanitized for personally identifiable information (PII), encrypted in transit and at rest, parsed into a strict Pydantic schema, and serialized for persistent storage with backups enabled. The minimum inputs are:
	•	business_name (string, required) — definitive string match.
	•	address (string, required) — definitive string match, though its utility may be limited.
	•	registration_id (string, optional).
	•	country_code (ISO2, required) — definitive string match.
	•	website_url (string, optional, treated as PII).
	•	documents (binary, optional, treated as PII) — e.g., shareholder lists or registry extracts.

## Risks and Trade-offs

| **Risk** | **Likelihood** | **Impact** | **Mitigation Strategy** |
|----------|----------------|------------|--------------------------|
| **Data provenance & legal admissibility**: Third-party documents (e.g. shareholder lists, registry extracts) may be processed without clear licensing or consent. | Medium | High (GDPR/DPA non-compliance, contractual disputes) | Require provenance metadata at ingestion. Store source, license, and timestamp with each document. Integrate license/consent checks into Pydantic schema validation. |
| **Audit trail insufficiency**: Logs may not meet evidentiary standards (tamper-proofing, immutability). | Medium | High (regulators dismiss logs; liability in disputes) | Use append-only, cryptographically signed audit logs (e.g. immudb, WORM storage). Automate retention and immutability guarantees. |
| **Explainability threshold**: Verdicts (ACCEPT/REJECT/REVIEW) may be seen as probabilistic or opaque. | High | High (FCA/AML regulators require deterministic reasoning) | Split decisions into deterministic core (sanctions, PEPs, KYC facts) and probabilistic layer (risk scoring). Generate structured “reason codes” alongside natural-language explanations. |
| **Model sufficiency (8B–14B tradeoff)**: Smaller models may be challenged as “underpowered” if errors occur. | Medium | Medium–High (litigation or reputational damage) | Document model selection process. Maintain benchmarking data showing 8B models meet required accuracy. Build process for escalating to larger models if benchmarks fall below threshold. |
| **Monitoring not regulator-friendly**: Metrics (Prometheus/Grafana) may be unreadable to auditors. | High | Medium | Build compliance dashboards that export metrics into human-readable evidence packs (PDF/CSV). Provide drill-down from job ID → metrics → decision lineage. |
| **Open-source reliance liability**: Security flaws, bias, or regressions in OSS dependencies. | Medium | Medium | Vendor-style governance: pin dependency versions, run regular SAST/DAST scans, monitor CVEs, and document fallback options. Assign ownership for OSS update review. |
| **Cloud-agnostic complexity**: Running across Vast.ai/RunPod/Nebius vs AWS/GCP may create compliance inconsistencies. | Low–Medium | Medium | Standardize deployment configs via Kubernetes. Enforce identical compliance controls across environments. Maintain provider risk assessments. |
| **Human feedback bottleneck**: Reliance on 500–1000 real-world feedback jobs per week may be insufficient or delayed. | Medium | Medium | Supplement with synthetic data, red-teaming, and periodic external audits. Set KPI that system cannot deploy without minimum volume of fresh feedback. |
| **Fine-tuning data scarcity**: Human evaluator data (500–1000 cases/week) may not scale or may be biased. | Medium | High (overfitting, biased models, regulatory rejection) | Curate evaluator pool. Enforce annotation standards. Store structured metadata with all training examples. Build process for independent validation of fine-tuned models. |
| **Vector-based fuzzy matching**: Use of embeddings for name matching may introduce false positives/negatives. | Medium | Medium | Combine deterministic string rules with vector similarity. Log and explain every match decision. Regularly recalibrate thresholds with ground truth data. |
| **GraphDB for entity linking**: Graph databases introduce governance and complexity risks. | Low–Medium | Medium–High (explainability gains vs infra overhead) | Limit scope to high-value cases (sanctions, PEP linkages). Document schema and graph logic. Ensure every graph edge is backed by verifiable evidence. |
| **GraphRAG explainability**: Graph-augmented RAG may generate reasoning chains that are persuasive but not legally admissible. | Medium | High | Constrain outputs: separate factual edges from model-inferred ones. Require human sign-off before graph-augmented verdicts are used in production. |
| **MCP server as core service**: Reliance on MCP introduces interoperability and governance risk. | Low | Medium | Document API contracts and lifecycle. Maintain fallbacks to direct toolchains. Ensure MCP service is independently monitored and audited. |
| **Cost analysis blind spots**: Cloud cost assumptions may miss hidden fees (egress, storage, network). | Medium | Medium | Run controlled cost-comparison pilots (AWS/GCP vs Vast.ai/RunPod). Document total cost of ownership (TCO). Include compliance/security overheads in analysis. |
| **Web search integration**: Extending to Google/Perplexity-style search introduces IP, copyright, and consent risks. | High | High (GDPR, IP law, defamation risk if misattributed) | Filter sources at ingestion. Store provenance metadata. Restrict use to fact-verifiable domains (official registers, licensed APIs). Maintain legal review of third-party ingestion. |
⸻

=======
# KYC Compliance - GenAI Agent Architecture

## Design Philosophy

### Traceability Priority
- **Human Process Fidelity**: Data flow must mirror human KYC officer dependency chain (identity → risk assessment → screening → investigation → documentation)
- **Chain of Custody**: Each step maps directly to human KYC officer activities, making audit trails intuitive for regulators and stakeholders
- **Process Transparency**: Stakeholders must be able to understand AI decisions because they follow familiar human workflows

### Explainability Priority
- **Natural Language Reasoning** > **Algorithmic Complexity**: A GenAI system that says "Found potential match for 'Mohammad Smith' considering common spelling variations" is infinitely more explainable than a black-box Levenshtein distance algorithm with 20 years of accumulated edge-case rules
- **Human-Readable Logic**: Prompts that mirror human thought processes ("Does this person appear on sanctions lists?") vs. cryptic algorithmic thresholds
- **Contextual Understanding**: GenAI can explain *why* it flagged something in plain language, not just *that* it was flagged

### Architecture 
- **GenAI-Native Design**: Build architectures that leverage GenAI's reasoning, context understanding, and natural language capabilities from the ground up
- **Anti-Pattern**: OpenAI wrapper functions bolted onto legacy ETL pipelines (30 Lambda step functions triggered by Glue) = using smartphones only for phone calls
- **Transformative Patterns**: Ray clusters with vLLM for distributed reasoning, GraphRAG for contextual knowledge synthesis, agent-based workflows that mirror human cognitive patterns

[![Docker Hub](https://img.shields.io/docker/pulls/michaelsigamani/proj-grounded-telescopes)](https://hub.docker.com/r/michaelsigamani/proj-grounded-telescopes)

### Run:
```bash
docker compose -f compose.dev.yaml up --build
docker compose -f compose.prod.yaml up
```

