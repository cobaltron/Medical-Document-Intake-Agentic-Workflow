from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from agno.media import Image
from agno.workflow import Workflow, Step, StepInput, StepOutput

from agents.ingestion_agent import ingestion_agent, IngestionResult
from agents.extraction_agent import extraction_agent
from agents.normalization_agent import normalization_agent
from agents.output_agent import output_agent


# ── Utility ───────────────────────────────────────────────────────────────────

def load_image(path: str) -> Image:
    with open(path, "rb") as f:
        return Image(content=f.read())

def strip_json_fences(text: str) -> str:
    """Strip markdown code fences that Pixtral sometimes wraps output in."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ── Step executors ────────────────────────────────────────────────────────────

def ingestion_step(step_input: StepInput) -> StepOutput:
    """Stage 1: Detect document type from image."""
    print("\nStage 1: Detecting document type...")

    image = step_input.additional_data.get("image")

    response = ingestion_agent.run(
        "What type of medical document is this?",
        images=[image]
    )
    result: IngestionResult = response.content

    print(f"  Doc Type   : {result.doc_type}")
    print(f"  Confidence : {result.confidence}")
    print(f"  Notes      : {result.notes}")

    # Pass result forward as dict so next step can read it
    return StepOutput(content=result.model_dump())


def extraction_step(step_input: StepInput) -> StepOutput:
    """Stage 2: Extract fields from document image."""
    print("\nStage 2: Extracting fields from document...")

    image = step_input.additional_data.get("image")
    ingestion_data = step_input.previous_step_content  # dict from Stage 1

    prompt = (
        f"This is a {ingestion_data['doc_type']} medical document. "
        f"Additional context: {ingestion_data['notes']}. "
        f"Extract all relevant fields and return as a clean JSON object only. "
        f"For fields you cannot read clearly, set value to null and list "
        f"the field name in a 'low_confidence_fields' array."
    )

    response = extraction_agent.run(prompt, images=[image])
    extracted = json.loads(strip_json_fences(response.content))
    extracted["doc_type"] = ingestion_data["doc_type"]  # carry doc_type forward

    print(f"  Extracted  : {json.dumps(extracted, indent=4)}")

    return StepOutput(content=extracted)


def normalization_step(step_input: StepInput) -> StepOutput:
    """Stage 3: Normalize and standardize extracted fields."""
    print("\n Stage 3: Normalizing extracted data...")

    extracted_data = step_input.previous_step_content  # dict from Stage 2

    response = normalization_agent.run(
        f"Normalize this extracted medical document data:\n"
        f"{json.dumps(extracted_data, indent=2)}"
    )
    result = response.content

    print(f"  Normalized Fields : {json.dumps(result.normalized_fields, indent=4)}")
    print(f"  Low Confidence    : {result.low_confidence_fields}")
    print(f"  Notes             :")
    for note in result.normalization_notes:
        print(f"    - {note}")

    return StepOutput(content=result.model_dump())


def output_step(step_input: StepInput) -> StepOutput:
    """Stage 4: Generate output and FHIR record."""
    print("\nStage 4: Generating final output and FHIR record...")

    normalized_data = step_input.previous_step_content  # dict from Stage 3
    
    # Generate a mock validation result since validation agent doesn't exist yet
    validation_status = "passed"
    low_confidence_flags = normalized_data.get("low_confidence_fields", [])
    if low_confidence_flags:
        validation_status = "needs_review"

    input_data = {
        "normalized_data": normalized_data,
        "validation_result": {
            "status": validation_status,
            "missing_fields": [],
            "inconsistencies": [],
            "low_confidence_flags": low_confidence_flags
        }
    }

    response = output_agent.run(
        f"Generate output based on this data:\n{json.dumps(input_data, indent=2)}"
    )
    result = response.content

    if isinstance(result, str):
        print(f"  Warning: Output agent returned string instead of Pydantic model: {result[:500]}")
        return StepOutput(content={"error": result})

    print(f"  Summary         : {result.summary}")
    print(f"  FHIR ID         : {result.fhir_record.id if result.fhir_record else 'N/A'}")
    print(f"  FHIR Type       : {result.fhir_record.resourceType if result.fhir_record else 'N/A'}")
    print(f"  Review Required : {result.review_required}")
    print(f"  Confidence      : {result.confidence_score}")

    return StepOutput(content=result.model_dump())


# ── Workflow definition ───────────────────────────────────────────────────────

medical_intake_workflow = Workflow(
    name="Medical Document Intake Pipeline",
    steps=[
        Step(name="Ingestion",     executor=ingestion_step),
        Step(name="Extraction",    executor=extraction_step),
        Step(name="Normalization", executor=normalization_step),
        Step(name="Output",        executor=output_step),
    ]
)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    image = load_image("sample_data/images.png")

    print("\n========================================")
    print(" Medical Document Intake Pipeline")
    print("========================================")

    medical_intake_workflow.run(
        message="Process this medical document.",
        additional_data={"image": image}   # image passed through all steps via additional_data
    )

    print("\n========================================")
    print(" Pipeline Complete")
    print("========================================\n")