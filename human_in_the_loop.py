import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Tools & LLMs Initialization
# ==========================================

# Tavily search tool added so the writer can look up current information/trends
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# Writer LLM bound with tools
writer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)

# ==========================================
# State Definition
# ==========================================

class State(TypedDict):
    topic: str 
    messages: Annotated[list, add_messages]
    draft: str 
    review_feedback: str
    is_approved: bool 
    attempt: int

# ==========================================
# Prompts
# ==========================================

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Write engaging, professional "
    "LinkedIn posts about the given topic. If the topic requires up-to-date "
    "information, statistics, or current trends, use the web search tool to "
    "gather fresh context before writing. "
    "Rules: strong hook in the first line, one clear takeaway, easy to skim "
    "with short paragraphs, roughly 150-200 words, end with an engaging "
    "question or CTA, no hashtags. "
    "If you receive feedback on a previous draft, address every point carefully."
)

# ==========================================
# Nodes
# ==========================================

def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    attempt = state.get("attempt", 0) + 1 
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    print(f"\n[Attempt {attempt}] Writer is drafting the post...")

    if attempt == 1:
        user_message = f"Write a LinkedIn post on this topic: {topic}. If you need current info, search the web first."
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected.\n\n"
            f"Reviewer feedback:\n{previous_feedback}\n\n"
            f"Write a NEW improved LinkedIn post that fixes every issue mentioned."
        )

    # CHANGED: Included past message history (`state["messages"]`) so the model 
    # retains memory of tool outputs across iterative loops instead of starting fresh.
    messages = [("system", WRITER_SYSTEM_PROMPT)] + state["messages"] + [("human", user_message)]
    
    # CHANGED: Used `writer_llm_with_tools` instead of base `writer_llm`
    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "attempt": attempt
    }

# NEW: ToolNode to execute search queries requested by the writer
tool_node = ToolNode(tools)

def extract_draft_node(state: State) -> dict:
    """NEW: Extracts final prose from the AI message once tool calls are finished."""
    last_message = state['messages'][-1]
    draft = last_message.content 
    print(f"\n[Draft ready]\n{'-' * 55}\n{draft}\n{'-' * 55}")
    return {"draft": draft}


def human_review_node(state: State) -> dict:
    """Pauses the graph using interrupt() and waits for human input."""
    print(f"\n[Reached human review — Attempt {state['attempt']}]")

    human_response = interrupt({
        "draft": state["draft"],
        "attempt": state["attempt"],
        "instruction": "Type 'approved' to accept, or type your feedback to request a rewrite."
    })

    response = human_response.strip()

    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {
            "is_approved": True,
            "review_feedback": "Approved by human."
        }
    else:
        return {
            "is_approved": False,
            "review_feedback": response
        }

# ==========================================
# Router Functions
# ==========================================

# NEW: Determines if the LLM wants to call a search tool or is ready to extract text
def should_use_tool(state: State):
    last_message = state['messages'][-1]
    if getattr(last_message, 'tool_calls', None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state: State):
    if state['is_approved']:
        print("\n[Post approved by human. Ending workflow.]")
        return END
    if state['attempt'] >= 3:
        print("\n[Reached max 3 attempts. Ending with last draft.]")
        return END 
    print(f"\n[Rejected. Looping back to writer for attempt {state['attempt'] + 1}...]")
    return "writer"

# ==========================================
# Graph Assembly
# ==========================================

graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)               # NEW
graph.add_node("extract_draft", extract_draft_node) # NEW
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "writer")

# CHANGED: Added conditional routing from writer to either tools or draft extraction
graph.add_conditional_edges(
    "writer",
    should_use_tool,
)

# NEW: Route tool outputs back to the writer so it can incorporate search results
graph.add_edge("tools", "writer")

# NEW: Send extracted draft text to human review
graph.add_edge("extract_draft", "human_review")

graph.add_conditional_edges(
    "human_review",
    should_stop_looping,
    {
        "writer": "writer",
        END: END,
    },
)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ==========================================
# Execution Loop
# ==========================================

if __name__ == "__main__":
    print("=" * 55)
    print("Welcome to the LinkedIn Post Generator (HITL + Tools Edition)")
    print("=" * 55)
    print("\nThis tool will search the web if needed, draft a post,")
    print("and pause for YOU to review and provide feedback.")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

    if not topic:
        print("\nNo topic given. Exiting.")
    else:
        print("\nStarting generation...\n")

        config = {"configurable": {"thread_id": "linkedin_session_1"}}

        initial_state = {
            "topic": topic,
            "messages": [],
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
        }

        result = app.invoke(initial_state, config=config)

        # Handles human-in-the-loop interrupts
        while "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0].value

            print("\n" + "=" * 55)
            print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
            print("=" * 55)
            print(interrupt_data["draft"])
            print("=" * 55)
            print(f"\n{interrupt_data['instruction']}")

            human_input = input("\nYour response: ").strip()

            result = app.invoke(Command(resume=human_input), config=config)

        print("\n" + "=" * 55)
        print("FINAL LINKEDIN POST")
        print("=" * 55)
        print(result["draft"])
        print("=" * 55)
        print(f"Total attempts: {result['attempt']}")
        print(f"Approved by human: {result['is_approved']}")