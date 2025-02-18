import os
import re
import io
import asyncio
import platform
import base64
from dotenv import load_dotenv
from typing import List, Optional
from typing_extensions import TypedDict
from playwright.async_api import Page, async_playwright
from PIL import Image
from langchain import hub
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import chain as chain_decorator, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load the environment variables from .env file
load_dotenv()

# Bounding box from browser annotation function
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

# The Agent's output
class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

# This represents the state of the agent as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # Set of marks (the bounding boxes from the browser annotation function)
    prediction: Prediction  # The Agent's output
    scratchpad: List[BaseMessage]  # A system message(s) containing the intermediate steps
    observation: str  # The most recent response from a tool

# The agent has 6 tools:
# - Click (at labeled box)
# - Type
# - Scroll
# - Wait
# - Go back
# - Go to search engine (Google)

async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (f"Failed to type in element from bounding box labeled as number {type_args}")
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."


# JS to run on each step to take a screenshot of the page, select the elements to annotate, and add bounding boxes
with open("mark_page.js") as f:
    mark_page_script = f.read()

# Annotate all buttons, inputs, text areas, etc. with numbered bounding boxes
@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            asyncio.sleep(3)
    screenshot = await page.screenshot()
    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip().rstrip(";")
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}

prompt = hub.pull("wfh/web-voyager")

llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

# After a tool is invoked, we want to update the scratchpad so the agent is aware of its previous steps
def update_scratchpad(state: AgentState):
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}

# Compose everything into a graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        # The lambda ensures the function's string output is mapped to the "observation" key in the AgentState
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
        )
    # Always return to the agent (by means of the update-scratchpad node)
    graph_builder.add_edge(node_name, "update_scratchpad")


# Any time the agent completes, this function is called to route the output to a tool or to the end user
def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)

graph = graph_builder.compile()

# print out executed steps and save intermediate screenshots
async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    async for event in event_stream:
        # We'll display an event stream here
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")

        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        print("\n".join(steps))

        # Save image to disk:
        image_data = base64.b64decode(event["agent"]["img"])
        img = Image.open(io.BytesIO(image_data))
        img.save(f"screenshot_{len(steps)}.png")  # Save with step number
        print(f"Screenshot saved to screenshot_{len(steps)}.png")

        if "ANSWER" in action:
            final_answer = action_input[0]
            break
    return final_answer


async def main():
    print("Starting browser...")
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    _ = await page.goto("https://www.google.com")

    question = "Explain the WebVoyager paper on arxiv"

    res = await call_agent(question, page)
    print(f"Final response: {res}")

if __name__ == "__main__":
    asyncio.run(main())
