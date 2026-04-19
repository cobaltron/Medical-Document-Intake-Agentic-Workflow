from typing import List

from agno.agent import Agent
from agno.media import Image
from agno.models.mistral.mistral import MistralChat
from pydantic import BaseModel

extraction_agent = Agent(
    model=MistralChat(id="pixtral-12b-2409"),
    instructions=[
        "You receive a medical document image and its detected type.",
        "Extract all relevant fields based on the document type.",
        "For fields you cannot read clearly, set value to null and add to a 'low_confidence_fields' list.",
        "Never guess or hallucinate field values.",
        "Return extracted data as a clean JSON object only."
    ]
)
