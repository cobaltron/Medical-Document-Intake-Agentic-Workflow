from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class NormalizedDocument(BaseModel):
    doc_type: str
    normalized_fields: Dict[str, Any]   # cleaned, standardized fields
    low_confidence_fields: List[str]     # carried forward from extraction
    normalization_notes: List[str]       # what was changed and why


normalization_agent = Agent(
    name="Normalization Agent",
    model=Gemini(id="gemini-2.5-flash"),
    output_schema=NormalizedDocument,
    use_json_mode=True,
    instructions=[
        "You receive raw extracted fields from a medical document as JSON.",
        "You must return the COMPLETE normalized data in 'normalized_fields'.",
        "Every single field from the input must appear in 'normalized_fields' — do not skip any.",
        "",
        "Apply these normalization rules to the values inside 'normalized_fields':",
        "1. Dates: convert all formats to ISO 8601 (YYYY-MM-DD).",
        "2. Drug names: convert brand names to generic. E.g. 'Crocin' → 'Paracetamol', 'Dolo' → 'Paracetamol'.",
        "3. Dosages: standardize units. E.g. '500mg' → '500 mg', '.65g' → '650 mg'.",
        "4. Abbreviations: expand medical shorthand. "
           "'HTN' → 'Hypertension', 'DM2' → 'Type 2 Diabetes Mellitus', "
           "'OD' → 'Once daily', 'BD' → 'Twice daily', 'SOS' → 'As needed', "
           "'wks' → 'weeks', 'mths' → 'months'.",
        "5. Names: title-case all patient and doctor names.",
        "6. Null fields: keep them as null — do not infer or guess.",
        "",
        "In 'normalization_notes', log each change made. E.g. 'Crocin → Paracetamol (generic)'.",
        "In 'low_confidence_fields', carry forward the list from the input unchanged.",
        "",
        "IMPORTANT: normalized_fields must never be empty. It must contain all fields from the input.",

    ],
)