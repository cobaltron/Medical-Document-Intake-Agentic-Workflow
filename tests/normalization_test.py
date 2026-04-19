import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_normalization.py
from dotenv import load_dotenv
load_dotenv()

from agents.normalization_agent import normalization_agent
import json

# Simulated output from your extraction agent
mock_extraction = {
    "doc_type": "prescription",
    "extracted_fields": {
        "patient_name": "rajarshi Lahiri",
        "date": "12/03/25",
        "doctor_name": "dr. a. mehta",
        "medications": [
            {"name": "Crocin", "dose": "500mg", "frequency": "BD", "duration": "5 days"},
            {"name": "Dolo 650", "dose": ".65g", "frequency": "SOS", "duration": None},
        ],
        "diagnosis": "HTN, DM2",
        "follow_up": "2 wks"
    },
    "low_confidence_fields": ["follow_up"]
}

response = normalization_agent.run(
    f"Normalize this extracted medical document data:\n{json.dumps(mock_extraction, indent=2)}"
)

result = response.content

print(f"Doc Type            : {result.doc_type}")
print(f"Normalized Fields   : {json.dumps(result.normalized_fields, indent=2)}")
print(f"Low Confidence      : {result.low_confidence_fields}")
print(f"Normalization Notes : ")
for note in result.normalization_notes:
    print(f"  - {note}")