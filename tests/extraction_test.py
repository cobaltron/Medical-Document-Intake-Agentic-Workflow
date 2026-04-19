import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image                  # ← import Image from agno
from agents.extraction_agent import extraction_agent
from pathlib import Path
from agno.models.mistral.mistral import MistralChat

def load_image(path: str) -> Image:
    with open(path, "rb") as f:
        image_bytes = f.read()                # ← read as raw bytes, not base64
    return Image(content=image_bytes)         # ← wrap in agno Image object

def test_ingestion(image_path: str):
    print(f"\n--- Testing Extraction Agent ---")
    print(f"Document: {image_path}\n")

    image = load_image(image_path)

    response = extraction_agent.run(
        "What text is in this document?",
        images=[image]                        # ← pass Image object directly
    )

    result = response.content

    print(f"response : {result}")

if __name__ == "__main__":
    test_ingestion("../sample_data/images.png")