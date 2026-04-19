from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class FHIRCoding(BaseModel):
    system: str
    code: str
    display: str


class FHIRRecord(BaseModel):
    resourceType: str
    id: str
    status: str
    fields: Dict[str, Any]             # normalized fields mapped to FHIR structure


class OutputResult(BaseModel):
    fhir_record: FHIRRecord
    summary: str                       # 2-3 sentence plain English summary
    review_required: bool
    review_reasons: List[str]          # populated if review_required is True
    confidence_score: float            # 0.0 to 1.0 overall pipeline confidence


output_agent = Agent(
    name="Output Agent",
    model=Gemini(id="gemini-2.5-flash"),
    output_schema=OutputResult,
    use_json_mode=True,
    instructions=[
        "You receive normalized medical document data as JSON.",
        "Produce a structured output with four components:",
        "1. fhir_record: map the normalized fields into a FHIR-compatible structure.",
        "   Set resourceType based on doc_type:",
        "   - prescription       → MedicationRequest",
        "   - discharge_summary  → DocumentReference",
        "   - lab_report         → DiagnosticReport",
        "   - insurance_card     → Coverage",
        "   - referral_letter    → ServiceRequest",
        "   - vaccination_record → Immunization",
        "   Set id to a placeholder string 'generated-id'.",
        "   Set status to the validation status from the input.",
        "   Place all normalized fields under 'fields'.",
        "2. summary: write 2-3 plain English sentences summarizing the document.",
        "   If review is required, clearly state what needs attention.",
        "3. review_required: set to true if validation status is 'needs_review' or 'fail'.",
        "4. review_reasons: list the specific reasons review is needed.",
        "   Copy from missing_fields, inconsistencies, and low_confidence_flags in the validation result.",
        "5. confidence_score: a float from 0.0 to 1.0.",
        "   Start at 1.0 and subtract:",
        "   - 0.1 per missing required field",
        "   - 0.15 per inconsistency",
        "   - 0.05 per low confidence flag",
        "   Minimum score is 0.0.",
        "Never infer or hallucinate field values. Use only what is in the input.",
    ],
)