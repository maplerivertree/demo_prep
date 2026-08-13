"""
Cascade Home Services — Customer Inquiry Triage Agent
======================================================

Plain-English overview for a non-developer maintainer:

This script reads one customer inquiry (an email/web-form message saved as a
text file), sends it to an AI model hosted on Microsoft Foundry, and asks the
AI to:

  1. Figure out which of 4 categories the inquiry belongs to
     (NEW_JOB, SCHEDULING, COMPLAINT, BILLING)
  2. Write a short draft reply that a human can look over and send
  3. Decide whether a human needs to be alerted right away ("escalate")

The AI is only allowed to use the facts in the two files under /docs
(price_sheet.md and service-policy.md) when it talks about prices or
policies. We do this by literally pasting those two files into the
instructions we give the AI, so it never has to guess.

The full behavior (categories, escalation rules, output format) comes from
docs/SPEC.md — see that file if you want to change the agent's rules.

How to run this from a terminal:

    python src/agent.py sample-inquiries/new-job.txt

The result is printed to the terminal. Nothing is ever sent automatically —
a human always reviews the draft reply before it goes out.
"""

# --- Standard library imports -------------------------------------------
import asyncio  # lets us call the "async" AI functions from a normal script
import os  # lets us read environment variables (our Foundry settings)
import re  # lets us strip markdown code fences some models add around JSON
import sys  # lets us read the command-line argument (the inquiry file path)
from pathlib import Path  # an easy way to build file paths that work on any OS
from typing import Literal  # lets us restrict a field to a fixed set of values

# --- Third-party imports -------------------------------------------------
from dotenv import load_dotenv  # reads settings out of a local ".env" file
from pydantic import BaseModel, Field, ValidationError  # describes the exact shape of the AI's answer

# --- Microsoft Agent Framework imports -----------------------------------
# Agent: the thing that actually talks to the AI model and manages the conversation.
# FoundryChatClient: tells the Agent to use a Microsoft Foundry project as the AI backend.
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

# configure_otel_providers: turns on OpenTelemetry tracing that the Agent
# Framework understands out of the box (no manual span code needed) so you
# can watch each step of a run in the Foundry Toolkit's trace viewer.
from agent_framework.observability import configure_otel_providers

# AzureCliCredential: lets us log in to Azure using the same login as the
# "az login" command, instead of typing in any secret API keys.
from azure.identity import AzureCliCredential

# -------------------------------------------------------------------------
# Step 1: Load configuration
# -------------------------------------------------------------------------
# This reads a local ".env" file (if one exists) and copies its values into
# the environment, so os.environ.get(...) below can find them. See
# .env.sample for the two values this agent needs.
load_dotenv()

# -------------------------------------------------------------------------
# Step 1b: Turn on tracing.
# -------------------------------------------------------------------------
# This sends OpenTelemetry traces to the Foundry Toolkit's local trace
# collector, running on your machine, so you can open the trace viewer in
# VS Code and see exactly what the agent did on each run (which model was
# called, what was sent, what came back, how long it took, etc).
# enable_sensitive_data=True also captures the actual prompt/response text,
# which is what you want while developing locally.
configure_otel_providers(
    vs_code_extension_port=4317,  # Foundry Toolkit's local trace collector (gRPC)
    enable_sensitive_data=True,
)

# The folder that contains this script (src/), and its parent (the project root).
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DOCS_DIR = PROJECT_ROOT / "docs"


def load_grounding_documents() -> str:
    """Read the two documents the agent is allowed to use, and return their
    text glued together with clear headers, so we can paste it straight into
    the AI's instructions.
    """
    print_step("GROUNDING", f"Looking up {DOCS_DIR / 'price_sheet.md'}")
    price_sheet = (DOCS_DIR / "price_sheet.md").read_text(encoding="utf-8")

    print_step("GROUNDING", f"Looking up {DOCS_DIR / 'service-policy.md'}")
    service_policy = (DOCS_DIR / "service-policy.md").read_text(encoding="utf-8")

    return (
        "### price_sheet.md\n"
        f"{price_sheet}\n\n"
        "### service-policy.md\n"
        f"{service_policy}"
    )


# -------------------------------------------------------------------------
# Step 2: Describe the exact shape we want the AI's answer to come back in
# -------------------------------------------------------------------------
# Using a Pydantic model here means the AI framework will ask the model for
# JSON that matches these fields exactly, and will hand us back a Python
# object instead of us having to parse text ourselves.
class TriageResult(BaseModel):
    """The structured result the agent produces for one customer inquiry."""

    category: Literal["NEW_JOB", "SCHEDULING", "COMPLAINT", "BILLING"] = Field(
        description="Exactly one of the four inquiry categories."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident the agent is in the category it chose."
    )
    escalate: bool = Field(
        description="True if a human needs to look at this inquiry right away."
    )
    escalate_reason: str = Field(
        default="",
        description=(
            "One-line reason for escalating. Leave this empty when escalate is false."
        ),
    )
    draft_reply: str = Field(
        description="A short, warm, plain-spoken draft reply for a human to review and send."
    )


# -------------------------------------------------------------------------
# Step 3: Build the instructions we give the AI (this is where SPEC.md's
# rules live, in plain English, plus the two grounding documents pasted in)
# -------------------------------------------------------------------------
def build_system_instructions() -> str:
    grounding_documents = load_grounding_documents()

    return f"""
You are the customer-inquiry triage agent for Cascade Home Services, a
12-person plumbing and HVAC company in the greater Seattle area. A human
owner currently reads every inquiry personally; your job is to do the first
pass so they don't have to triage everything themselves.

For every inquiry you are given, you must do all of the following:

1. CLASSIFY the inquiry into exactly one of these four categories:
   - NEW_JOB: a request for new work or a price estimate.
   - SCHEDULING: a change, confirmation, or question about an existing appointment.
   - COMPLAINT: dissatisfaction with completed or ongoing work.
   - BILLING: a question about an invoice or payment.

2. GROUND yourself ONLY in the two documents below (a price sheet and a
   service-policy document). Never invent a price or policy. If the
   documents do not cover something the customer is asking about, say so
   plainly in the draft reply instead of guessing.

3. DRAFT a reply for a human to review and send. The reply must be:
   - warm and plain-spoken, and short.
   - never a promise of a specific appointment time — only a human
     coordinator schedules appointments.
   - for NEW_JOB inquiries, include the relevant prices from the price
     sheet, clearly labeled as estimates (not final quotes).

4. DECIDE whether to ESCALATE (set escalate to true) when any of these are true:
   - the inquiry is a COMPLAINT, or
   - the inquiry mentions active water damage, a gas smell, or any other
     safety risk, or
   - your own classification confidence is low.
   When you escalate, give a one-line reason in escalate_reason. When you do
   not escalate, leave escalate_reason as an empty string.

Remember: you never send anything yourself. Every draft_reply is only a
suggestion for a human to approve.

Here are the two documents you must use for all prices and policies:

{grounding_documents}
""".strip()


# -------------------------------------------------------------------------
# Step 3b: Some models don't return plain JSON — they wrap it in a markdown
# code fence like ```json ... ```. This helper strips that fence off (if it's
# there) so parsing works no matter which model is selected.
# -------------------------------------------------------------------------
CODE_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", flags=re.DOTALL)


def strip_markdown_code_fence(text: str) -> str:
    """Remove a surrounding ```...``` markdown code fence, if present.

    Example input:
        ```json
        {"category": "BILLING"}
        ```
    Returns:
        {"category": "BILLING"}

    If there is no code fence, the text is returned unchanged (just trimmed).
    """
    cleaned = text.strip()
    match = CODE_FENCE_PATTERN.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


# -------------------------------------------------------------------------
# Step 3c: A tiny helper so every step of the agent's work is printed to the
# terminal in the same, easy-to-scan format: "[STEP_NAME] message".
# -------------------------------------------------------------------------
def print_step(step_name: str, message: str) -> None:
    """Print one line of the step-by-step trace, e.g. [CLASSIFY] category: NEW_JOB."""
    print(f"[{step_name}] {message}")


# -------------------------------------------------------------------------
# Step 4: The main logic — read the inquiry file, call the AI, print the result
# -------------------------------------------------------------------------
async def triage_inquiry(inquiry_text: str) -> TriageResult:
    """Send one inquiry to the Foundry-hosted model and return the structured result."""

    # These two settings tell the agent which Foundry project and which
    # model deployment inside it to talk to. See .env.sample for details.
    project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    model_deployment_name = os.environ.get("FOUNDRY_MODEL_DEPLOYMENT_NAME")

    if not project_endpoint or not model_deployment_name:
        raise SystemExit(
            "Missing configuration. Please create a .env file (copy .env.sample) "
            "and set FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL_DEPLOYMENT_NAME."
        )

    # AzureCliCredential re-uses your existing "az login" session, so no API
    # keys or secrets are ever stored in this project.
    credential = AzureCliCredential()

    # The Agent is the object we actually talk to. Its "client" tells it to
    # use our Microsoft Foundry project and model deployment as the AI brain,
    # and "instructions" is the system prompt built above (SPEC.md rules +
    # the two grounding documents).
    agent = Agent(
        client=FoundryChatClient(
            project_endpoint=project_endpoint,
            model=model_deployment_name,
            credential=credential,
        ),
        name="cascade-triage-agent",
        instructions=build_system_instructions(),
    )

    # Ask the agent to respond, and request that the response be shaped
    # exactly like our TriageResult model (see Step 2 above).
    print_step("CLASSIFY", "Sending inquiry to the AI model for classification and drafting...")
    response = await agent.run(inquiry_text, options={"response_format": TriageResult})

    # Some models return clean JSON; others wrap it in a ```json ... ``` code
    # fence. Strip that off ourselves before parsing, so this works the same
    # way regardless of which model is selected.
    cleaned_text = strip_markdown_code_fence(response.text)

    if not cleaned_text:
        raise SystemExit(f"The agent did not return a structured result. Raw text:\n{response.text}")

    try:
        result = TriageResult.model_validate_json(cleaned_text)
    except ValidationError as exc:
        raise SystemExit(
            f"The agent's reply could not be parsed as the expected result: {exc}\n\nRaw text:\n{response.text}"
        )

    # --- Trace: show the classification the agent settled on -----------
    print_step("CLASSIFY", f"category={result.category}  confidence={result.confidence}")

    # --- Trace: show the escalation decision ----------------------------
    if result.escalate:
        print_step("ESCALATE", f"true — {result.escalate_reason}")
    else:
        print_step("ESCALATE", "false — no escalation needed")

    # --- Trace: note that the draft reply is ready ----------------------
    print_step("DRAFT", "Draft reply written, ready for human review.")

    return result


def print_result(result: TriageResult) -> None:
    """Print the triage result to the terminal in a simple, readable way."""
    print("category:     ", result.category)
    print("confidence:   ", result.confidence)
    print("escalate:     ", result.escalate, f"({result.escalate_reason})" if result.escalate else "")
    print("draft_reply:")
    print(result.draft_reply)


def main() -> None:
    # This script expects exactly one argument: the path to the inquiry file.
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/agent.py <path-to-inquiry-file>")

    inquiry_path = Path(sys.argv[1])
    if not inquiry_path.is_file():
        raise SystemExit(f"Inquiry file not found: {inquiry_path}")

    print_step("READ", f"Reading inquiry from {inquiry_path}")
    inquiry_text = inquiry_path.read_text(encoding="utf-8")

    result = asyncio.run(triage_inquiry(inquiry_text))
    print("-" * 40)
    print_result(result)


if __name__ == "__main__":
    main()
