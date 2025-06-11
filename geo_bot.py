import base64
import json
import re
from io import BytesIO
from typing import Tuple, List, Optional, Dict, Any, Type

from PIL import Image
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from mapcrunch_controller import MapCrunchController

# The "Golden" Prompt (v6): Combines clear mechanics with robust strategic principles.

AGENT_PROMPT_TEMPLATE = """
**Mission:** You are an expert geo-location agent. Your goal is to find clues to determine your location within a limited number of steps.

**Current Status:**
- **Remaining Steps: {remaining_steps}**
- **Available Actions This Turn: {available_actions}**

---
**Core Principles of an Expert Player:**

1.  **Navigate with Labels:** `MOVE_FORWARD` follows the green 'UP' arrow. `MOVE_BACKWARD` follows the red 'DOWN' arrow. These labels are your most reliable compass. If there are no arrows, you cannot move.

2.  **Efficient Exploration (to avoid "Bulldozer" mode):**
    - **Pan Before You Move:** At a new location or an intersection, it's often wise to use `PAN_LEFT` or `PAN_RIGHT` to quickly survey your surroundings before committing to a move.
    - **Don't Get Stuck:** If you've moved forward 2-3 times down a path and found nothing but repetitive scenery (like an empty forest or highway), consider it a barren path. It's smarter to turn around (using `PAN`) and check another direction.

3.  **Be Decisive:** If you find a truly definitive clue (like a full, readable address or a sign with a unique town name), `GUESS` immediately. Don't waste steps.

4.  **Final Step Rule:** If `remaining_steps` is **exactly 1**, your action **MUST be `GUESS`**.

---
**Context & Task:**
Analyze your full journey history and current view, apply the Core Principles, and decide your next action in the required JSON format.

**Action History:**
{history_text}

**JSON Output Format:**
Your response MUST be a valid JSON object wrapped in ```json ... ```.
- For exploration: `{{"reasoning": "...", "action_details": {{"action": "ACTION_NAME"}} }}`
- For the final guess: `{{"reasoning": "...", "action_details": {{"action": "GUESS", "lat": <float>, "lon": <float>}} }}`
"""

BENCHMARK_PROMPT = """
Analyze the image and determine its geographic coordinates.
1.  Describe visual clues.
2.  Suggest potential regions.
3.  State your most probable location.
4.  Provide coordinates in the last line in this exact format: `Lat: XX.XXXX, Lon: XX.XXXX`
"""


class GeoBot:
    def __init__(
        self,
        model: Type,
        model_name: str,
        use_selenium: bool = True,
        headless: bool = False,
    ):
        self.model: BaseChatModel = model(model=model_name)
        self.model_name = model_name
        self.use_selenium = use_selenium
        self.controller = MapCrunchController(headless=headless)

    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.thumbnail((1024, 1024))
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _create_message_with_history(
        self, prompt: str, image_b64_list: List[str]
    ) -> List[HumanMessage]:
        """Creates a message for the LLM that includes text and a sequence of images."""
        content = [{"type": "text", "text": prompt}]
        # Add the JSON format instructions right after the main prompt text
        content.append(
            {
                "type": "text",
                "text": '\n**JSON Output Format:**\nYour response MUST be a valid JSON object wrapped in ```json ... ```.\n- For exploration: `{{"reasoning": "...", "action_details": {{"action": "ACTION_NAME"}} }}`\n- For the final guess: `{{"reasoning": "...", "action_details": {{"action": "GUESS", "lat": <float>, "lon": <float>}} }}`',
            }
        )

        for b64_string in image_b64_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_string}"},
                }
            )
        return [HumanMessage(content=content)]

    def _create_llm_message(self, prompt: str, image_b64: str) -> List[HumanMessage]:
        """Original method for single-image analysis (benchmark)."""
        return [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ]
            )
        ]

    def _parse_agent_response(self, response: BaseMessage) -> Optional[Dict[str, Any]]:
        """
        Robustly parses JSON from the LLM response, handling markdown code blocks.
        """
        try:
            assert isinstance(response.content, str), "Response content is not a string"
            content = response.content.strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = content
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Invalid JSON from LLM: {e}\nFull response was:\n{response.content}")
            return None

    def run_agent_loop(self, max_steps: int = 10) -> Optional[Tuple[float, float]]:
        history: List[Dict[str, Any]] = []

        for step in range(max_steps, 0, -1):
            print(f"\n--- Step {max_steps - step + 1}/{max_steps} ---")

            self.controller.setup_clean_environment()

            self.controller.label_arrows_on_screen()

            screenshot_bytes = self.controller.take_street_view_screenshot()
            if not screenshot_bytes:
                print("Failed to take screenshot. Ending agent loop.")
                return None

            current_screenshot_b64 = self.pil_to_base64(
                image=Image.open(BytesIO(screenshot_bytes))
            )
            available_actions = self.controller.get_available_actions()
            print(f"Available actions: {available_actions}")

            history_text: str = ""
            image_b64_for_prompt: List[str] = []
            if not history:
                history_text = "No history yet. This is the first step."
            else:
                for i, h in enumerate(history):
                    history_text += f"--- History Step {i + 1} ---\n"
                    history_text += f"Reasoning: {h.get('reasoning', 'N/A')}\n"
                    history_text += f"Action: {h.get('action_details', {}).get('action', 'N/A')}\n\n"
                    image_b64_for_prompt.append(h["screenshot_b64"])

            image_b64_for_prompt.append(current_screenshot_b64)

            prompt = AGENT_PROMPT_TEMPLATE.format(
                remaining_steps=step,
                history_text=history_text,
                available_actions=json.dumps(available_actions),
            )

            message = self._create_message_with_history(prompt, image_b64_for_prompt)
            response = self.model.invoke(message)

            decision = self._parse_agent_response(response)

            if not decision:
                print(
                    "Response parsing failed. Using default recovery action: PAN_RIGHT."
                )
                decision = {
                    "reasoning": "Recovery due to parsing failure.",
                    "action_details": {"action": "PAN_RIGHT"},
                }

            decision["screenshot_b64"] = current_screenshot_b64
            history.append(decision)

            action_details = decision.get("action_details", {})
            action = action_details.get("action")
            print(f"AI Reasoning: {decision.get('reasoning', 'N/A')}")
            print(f"AI Action: {action}")

            if action == "GUESS":
                lat, lon = action_details.get("lat"), action_details.get("lon")
                if lat is not None and lon is not None:
                    return lat, lon
            elif action == "MOVE_FORWARD":
                self.controller.move("forward")
            elif action == "MOVE_BACKWARD":
                self.controller.move("backward")
            elif action == "PAN_LEFT":
                self.controller.pan_view("left")
            elif action == "PAN_RIGHT":
                self.controller.pan_view("right")

        print("Max steps reached. Agent did not make a final guess.")
        return None

    def analyze_image(self, image: Image.Image) -> Optional[Tuple[float, float]]:
        image_b64 = self.pil_to_base64(image)
        message = self._create_llm_message(BENCHMARK_PROMPT, image_b64)
        response = self.model.invoke(message)
        print(f"\nLLM Response:\n{response.content}")

        content = response.content.strip()
        last_line = ""
        for line in reversed(content.split("\n")):
            if "lat" in line.lower() and "lon" in line.lower():
                last_line = line
                break
        if not last_line:
            return None

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", last_line)
        if len(numbers) < 2:
            return None

        lat, lon = float(numbers[0]), float(numbers[1])
        return lat, lon

    def take_screenshot(self) -> Optional[Image.Image]:
        screenshot_bytes = self.controller.take_street_view_screenshot()
        if screenshot_bytes:
            return Image.open(BytesIO(screenshot_bytes))
        return None

    def close(self):
        if self.controller:
            self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
