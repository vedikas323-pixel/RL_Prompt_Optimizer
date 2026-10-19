"""Prompt Builder: expand a brief task description into a detailed, structured prompt.

The builder is purely local — no API calls, no target model involvement.
The target model (Claude / OpenAI / Gemini) only receives and executes the
finished prompt; it plays no part in writing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Task type registry
# ---------------------------------------------------------------------------

TASK_TYPES = [
    "Academic Research Report (B.Tech / M.Tech / PhD)",
    "Literature Review",
    "Code / Software Project",
    "Data Analysis Report",
    "Business / Strategy Report",
    "Explain a Concept (Tutorial)",
    "Custom (free-form template)",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BuildRequest:
    task_type: str
    title: str
    domain: str
    brief: str
    target_model: str          # "Claude" | "OpenAI" | "Gemini"
    # Academic extras
    degree: str = "B.Tech / B.E. Capstone"
    style: str = "IEEE"
    chapters: str = "1,2,3,4,5"
    extra_sections: str = ""
    word_count: str = "80–120 pages"
    # Code extras
    language: str = ""
    framework: str = ""
    features: str = ""
    # Analysis extras
    dataset: str = ""
    metrics: str = ""
    # Custom
    custom_template: str = ""


# ---------------------------------------------------------------------------
# Model-specific wrappers
# ---------------------------------------------------------------------------

def _wrap(body: str, target_model: str, title: str) -> str:
    if target_model == "Claude":
        return (
            f"<role>You are an expert academic writer, researcher, and domain specialist "
            f"in {title}.</role>\n\n"
            f"<task>\n{body}\n</task>\n\n"
            "<requirements>\n"
            "- Use formal, precise academic language throughout.\n"
            "- Explain every technical concept before using it.\n"
            "- Maintain logical flow and smooth transitions between sections.\n"
            "- Do not fabricate citations; use numbered placeholders [1], [2], etc.\n"
            "- Present tables in markdown format.\n"
            "- Use numbered headings (1.0, 1.1, 1.2 …).\n"
            "</requirements>"
        )
    if target_model == "Gemini":
        return (
            f"**Role:** Expert academic writer and domain specialist in {title}.\n\n"
            f"**Task:**\n{body}\n\n"
            "**Requirements:**\n"
            "- Formal academic language.\n"
            "- Explain every technical concept before using it.\n"
            "- Logical flow with smooth transitions.\n"
            "- No fabricated citations — use placeholders [1], [2], etc.\n"
            "- Tables in markdown format.\n"
            "- Numbered headings (1.0, 1.1, 1.2 …).\n"
        )
    # OpenAI default
    return (
        f"Act as an expert academic writer and domain specialist in {title}.\n\n"
        f"{body}\n\n"
        "Writing requirements:\n"
        "- Use formal academic language.\n"
        "- Explain every technical concept before using it.\n"
        "- Ensure logical flow and smooth transitions.\n"
        "- Do not fabricate citations; use numbered placeholders [1], [2], etc.\n"
        "- Present tables in markdown format.\n"
        "- Use numbered headings (1.0, 1.1, 1.2 …).\n"
    )


# ---------------------------------------------------------------------------
# Template builders per task type
# ---------------------------------------------------------------------------

def _academic_report(req: BuildRequest) -> str:
    chapter_nums = [c.strip() for c in req.chapters.split(",") if c.strip()]
    chapter_map = {
        "1": _ch1_intro,
        "2": _ch2_lit,
        "3": _ch3_method,
        "4": _ch4_results,
        "5": _ch5_conclusion,
    }
    extra = req.extra_sections.strip()
    chapters_text = "\n\n".join(
        chapter_map[n](req) for n in chapter_nums if n in chapter_map
    )
    body = f"""Write a professional research report suitable for a {req.degree} project.

Project Title: "{req.title}"
Domain: {req.domain}
Context: {req.brief}
Writing style: {req.style}-style academic writing with formal language, technical depth, and logical flow.
Target length: approximately {req.word_count} when formatted in Microsoft Word.

The report must include:
- Abstract (300–500 words)
- Keywords
- List of Figures (placeholders)
- List of Tables (placeholders)
- References ({req.style} format placeholders, citing papers from 2019–2025 from IEEE, Springer, Elsevier, ACM, Nature, or ScienceDirect)
- Appendices and Acronyms

{chapters_text}
"""
    if extra:
        body += f"\nAdditional sections required:\n{extra}"
    return body


def _ch1_intro(req: BuildRequest) -> str:
    return f"""# Chapter 1 – Introduction

Include ALL of the following sections with full detail:

1.1 Introduction — open with the broader context of {req.domain}.
1.2 Background of the Study — history and evolution of the problem space.
1.3 Problem Statement — a single precise sentence naming the core variables, followed by a paragraph on:
    - Data fragmentation and lack of interoperability
    - Limited predictive analytics
    - Difficulty in real-time decision-making
    - Need for explainable and autonomous AI
1.4 Research Gap — what existing work has NOT addressed.
1.5 Research Objectives — 6–8 measurable objectives tied directly to the problem.
1.6 Research Questions — 4–6 questions, each resolvable by the methodology.
1.7 Scope of the Study — boundaries: geography, datasets, algorithms, time frame.
1.8 Significance of the Study — impact on industry, policy, and academia.
1.9 Expected Outcomes — concrete deliverables.
1.10 Chapter Summary — one paragraph summarising the chapter."""


def _ch2_lit(req: BuildRequest) -> str:
    return f"""# Chapter 2 – Literature Review

Write a comprehensive literature review for the topic: "{req.title}".

Include ALL of the following sections:

2.1 Introduction — purpose and structure of the review.
2.2 Historical Evolution — how the field of {req.domain} has developed.
2.3 Key Theoretical Frameworks — foundational theories underpinning the work.
2.4 Machine Learning and AI Applications in this domain.
2.5 Deep Learning and Advanced Models.
2.6 Explainable AI (XAI) — methods such as SHAP, LIME, attention mechanisms.
2.7 Large Language Models in scientific data analysis.
2.8 Agentic AI and Multi-Agent Systems.
2.9 Data Integration and Interoperability.
2.10 Predictive Analytics and Anomaly Detection.
2.11 Existing Research Papers — summarise at least 15–20 relevant papers, each with: authors, year, method, dataset used, key finding, and limitation.
2.12 Comparative Analysis Table (markdown):
    Columns: Authors | Year | Method | Dataset | Advantages | Limitations
2.13 Research Gap Analysis — synthesise the gaps found across reviewed works and show precisely how the proposed system addresses them.
2.14 Literature Summary — one paragraph synthesising all themes."""


def _ch3_method(req: BuildRequest) -> str:
    return f"""# Chapter 3 – Methodology

Describe the complete research methodology for: "{req.title}".

Include ALL of the following sections:

3.1 Research Design — quantitative, AI-based experimental design; justify the choice.
3.2 System Architecture Overview — describe the end-to-end architecture with a placeholder diagram.
3.3 Data Collection Strategy — sources, formats, access methods.
3.4 Data Sources — list each source with URL/API, variables available, and format (CSV/NetCDF/JSON/REST).
3.5 Data Integration Pipeline — how heterogeneous sources are merged; ETL workflow.
3.6 Data Cleaning and Preprocessing — missing values, outliers, normalisation, resampling.
3.7 Feature Engineering — domain-specific features created, dimensionality reduction.
3.8 Database Design — schema, tables, relationships; ER diagram placeholder.
3.9 Machine Learning Pipeline — training / validation / test split, cross-validation strategy.
3.10 Algorithms Used — for each algorithm explain: working principle, hyperparameters tuned, and why it was chosen.
3.11 Agentic AI Architecture — list each agent, its responsibility, inputs, outputs, and inter-agent communication protocol.
3.12 Technology Stack — languages, libraries, frameworks, cloud/local deployment.
3.13 Evaluation Metrics — define each metric mathematically (Accuracy, Precision, Recall, F1, RMSE, MAE).
3.14 Experimental Setup — hardware, software versions, reproducibility details.
3.15 Flowchart — placeholder for the end-to-end process flow.
3.16 Sequence Diagram — placeholder showing agent interactions.
3.17 Methodology Summary."""


def _ch4_results(req: BuildRequest) -> str:
    return f"""# Chapter 4 – Results and Findings

Present complete results for the project: "{req.title}".

Generate realistic sample results (clearly labelled as illustrative). Include ALL of the following:

4.1 Introduction — what this chapter covers.
4.2 Dataset Statistics — table with: dataset name, number of records, features, date range, missing value %.
4.3 Preprocessing Results — before/after comparison of data quality.
4.4 Model Training Results — training/validation loss curves description; convergence behaviour.
4.5 Prediction Results — tables with actual vs predicted values for key outputs.
4.6 Classification / Regression Performance Table (markdown):
    Columns: Model | Accuracy | Precision | Recall | F1 | RMSE | MAE
4.7 Comparative Model Performance — compare all models side by side; explain which performed best and why.
4.8 Anomaly Detection Results — detected anomalies with examples.
4.9 Explainable AI Outputs — SHAP summary plots described; top 5 features per prediction.
4.10 Visualisation Descriptions — describe 5–8 charts/dashboards (placeholder figures).
4.11 Chatbot / NL Query Examples — 5 sample questions and system answers.
4.12 Discussion of Findings — interpret every result in the context of the research questions.
4.13 Challenges Encountered — data quality issues, computational constraints, API limits.
4.14 Limitations — honest assessment of what the results cannot show.
4.15 Chapter Summary."""


def _ch5_conclusion(req: BuildRequest) -> str:
    return f"""# Chapter 5 – Conclusion and Future Work

Write the concluding chapter for: "{req.title}".

Include ALL of the following:

5.1 Conclusion — a substantive narrative (not a bullet list) that synthesises what was accomplished.
5.2 Summary of Research Objectives Achieved — map each objective (from Chapter 1) to a concrete result.
5.3 Contributions — original contributions to knowledge; distinguish from existing work.
5.4 Practical Applications — real-world deployment scenarios with named stakeholders.
5.5 Societal and Policy Impact — benefits for government agencies, industry, and conservation.
5.6 Limitations — honest critical assessment; what could not be tested or validated.
5.7 Future Scope — at least 6 directions: digital twins, real-time IoT, autonomous monitoring, satellite AI, generative AI, reinforcement learning; for each give a one-paragraph technical description.
5.8 Final Concluding Remark — one strong closing paragraph that ties the research problem to the achieved outcomes."""


def _literature_review(req: BuildRequest) -> str:
    body = f"""Write a comprehensive standalone literature review on:
Topic: "{req.title}"
Domain: {req.domain}
Context: {req.brief}

Structure:
1.0 Introduction — scope and objectives of the review.
2.0 Historical Background and Evolution.
3.0 Core Themes — identify 4–6 major themes; dedicate one subsection to each.
4.0 Key Studies — summarise 15–20 papers per theme: authors, year, method, dataset, finding, limitation.
5.0 Comparative Analysis Table (markdown): Authors | Year | Method | Dataset | Advantages | Limitations
6.0 Theoretical Frameworks — foundational models/theories used across the literature.
7.0 Research Gaps — synthesise what remains unresolved.
8.0 Summary — one paragraph consolidating all themes and gaps."""
    return body


def _code_project(req: BuildRequest) -> str:
    lang = req.language or "Python"
    framework = req.framework or "appropriate framework"
    features_list = req.features or req.brief
    body = f"""You are a senior software engineer and {req.domain} expert.

Project: "{req.title}"
Language: {lang}
Framework / Stack: {framework}
Core features required:
{features_list}

Deliver a complete, production-quality implementation including:
1. Architecture overview and component diagram (text description).
2. Complete source code with inline comments explaining every non-obvious decision.
3. Data models / schema definitions.
4. Unit tests for each major function (use {lang}'s standard testing library).
5. API documentation (docstrings or OpenAPI spec).
6. README with setup, usage, and example commands.
7. Error handling and input validation at all system boundaries.
8. Performance considerations and known limitations.

Requirements:
- Prioritise correctness and clarity over cleverness.
- No external dependencies unless justified.
- All edge cases handled explicitly."""
    return body


def _data_analysis(req: BuildRequest) -> str:
    body = f"""You are a senior data scientist and {req.domain} expert.

Task: "{req.title}"
Dataset: {req.dataset or req.brief}
Required metrics / outputs: {req.metrics or "accuracy, RMSE, feature importance, visualisations"}

Deliver a complete data analysis report including:
1. Executive Summary (200 words).
2. Dataset Description — shape, types, missing values, distributions.
3. Exploratory Data Analysis — describe 5–8 key visualisations.
4. Feature Engineering — transformations applied and rationale.
5. Model Selection — compare at least 3 algorithms with justification.
6. Training and Evaluation — cross-validation, metric tables.
7. Results — interpret every number; explain what it means for the domain.
8. Limitations and risks in the analysis.
9. Recommendations — 3–5 actionable conclusions for stakeholders.
10. Appendix — full code (well-commented)."""
    return body


def _business_report(req: BuildRequest) -> str:
    body = f"""You are a senior strategy consultant and {req.domain} expert.

Report: "{req.title}"
Context: {req.brief}

Write a professional business / strategy report including:
1. Executive Summary (250 words, standalone).
2. Background and Context.
3. Problem / Opportunity Definition.
4. Market / Environment Analysis (use PESTLE or SWOT as appropriate).
5. Options Considered — at least 3 alternatives with pros and cons.
6. Recommended Strategy — detailed rationale.
7. Implementation Roadmap — phases, timelines, resources.
8. Risk Assessment — likelihood / impact matrix (markdown table).
9. KPIs and Success Metrics.
10. Conclusion and Next Steps."""
    return body


def _concept_explainer(req: BuildRequest) -> str:
    body = f"""You are an expert educator in {req.domain}.

Topic: "{req.title}"
Audience context: {req.brief}

Write a comprehensive tutorial / explainer that:
1. Opens with a motivating real-world example before any definitions.
2. Defines every term precisely on first use.
3. Builds up from fundamentals: start simple, layer complexity gradually.
4. Includes worked examples with step-by-step reasoning for every concept.
5. Adds visual descriptions (diagrams as text/ASCII or described placeholders).
6. Includes a "Common Misconceptions" section.
7. Closes with a summary table: Concept | Definition | Key Formula / Rule | Example.
8. Suggests 3–5 further reading resources (placeholder citations)."""
    return body


def _custom(req: BuildRequest) -> str:
    tpl = req.custom_template.strip()
    if not tpl:
        return f"Task: {req.title}\nDomain: {req.domain}\nContext: {req.brief}"
    # Substitute simple placeholders
    return (tpl
            .replace("{title}", req.title)
            .replace("{domain}", req.domain)
            .replace("{brief}", req.brief))


_BUILDERS = {
    TASK_TYPES[0]: _academic_report,
    TASK_TYPES[1]: _literature_review,
    TASK_TYPES[2]: _code_project,
    TASK_TYPES[3]: _data_analysis,
    TASK_TYPES[4]: _business_report,
    TASK_TYPES[5]: _concept_explainer,
    TASK_TYPES[6]: _custom,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(req: BuildRequest) -> str:
    """Return the fully expanded, model-formatted prompt."""
    builder = _BUILDERS.get(req.task_type, _custom)
    body = builder(req)
    return _wrap(body, req.target_model, req.title)
