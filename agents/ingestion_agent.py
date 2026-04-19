from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel
from typing import Literal

class IngestionResult(BaseModel):
    doc_type: Literal[
        "insurance_card",
        "prescription",
        "discharge_summary",
        "lab_report",
        "referral_letter",
        "vaccination_record",
        "unknown"
    ]
    confidence: Literal["high", "medium", "low"]
    notes: str

ingestion_agent = Agent(
    name="Ingestion Agent",
    model=Gemini(id="gemini-2.5-flash"),
    output_schema=IngestionResult,
    instructions=[
        "You receive a medical document as an image.",
        "Your only job is to identify what TYPE of document it is.",
        "Choose from: insurance_card, prescription, discharge_summary, lab_report, referral_letter, vaccination_record, unknown.",
        "Set confidence based on how certain you are.",
        "In notes, flag anything relevant: poor image quality, handwritten content, non-English text, multiple documents in one image.",
        "Do not extract any fields — that is not your job.",
    ],
)