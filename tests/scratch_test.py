import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.google import Gemini

agent = Agent(
    name="Test Agent",
    model=Gemini(id="gemini-2.5-flash"),
    instructions=["Respond in one sentence only."],
)

agent.print_response("Are you working?")