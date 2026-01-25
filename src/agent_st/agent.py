from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
#from langchain.agents import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Literal, Optional, List, TypedDict
from pydantic import BaseModel, Field

load_dotenv()

lf = Langfuse()
langfuse_handler = CallbackHandler()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Token usage tracker
class TokenUsage:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def add_usage(self, response):
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)
    
    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def get_usage(self):
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }

token_usage = TokenUsage()

class ResumeSearchInput(BaseModel):
    """Input for resume search tool."""
    query: str = Field(description="Search query from user")
    section_filter: Optional[List[str]] = Field(default=None, description="List of sections to filter (e.g., ['EXPERIENCE', 'SKILLS'])")

@tool(args_schema=ResumeSearchInput)
def resume_search_tool(query: str, section_filter: Optional[List[str]] = None) -> str:
    """Retrieve relevant resume chunks based on the query.
    
    Apply section filter if provided to narrow down results.
    Returns chunk-level results with metadata.
    """
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )
    
    # Apply section filter if provided
    qdrant_filter = None
    if section_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="section",
                    match=MatchAny(any=section_filter)
                )
            ]
        )
    
    # Retrieve chunks with similarity search
    docs = vector_store.similarity_search_with_score(
        query, 
        k=10,
        filter=qdrant_filter
    )
    
    if docs:
        formatted_results = []
        for idx, (doc, score) in enumerate(docs):
            formatted_results.append(f"""
Chunk {idx + 1}:
- Resume ID: {doc.metadata.get('resume_id', 'N/A')}
- Category: {doc.metadata.get('category', 'N/A')}
- Section: {doc.metadata.get('section', 'N/A')}
- Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}
- Relevance Score: {score:.3f}
- Content: {doc.page_content}
""")
        
        context = "\n".join(formatted_results)
        return context
    return "No relevant documents found."

# Agent state definition
class AgentState(MessagesState):
    next_agent: str

# Supervisor prompt
supervisor_prompt = """You are a supervisor agent that routes user queries to specialized agents.

Your ONLY job is to analyze the user's intent and route to the appropriate agent:
- Use 'resume_specialist' for queries about experience, employment history, education background
- Use 'skill_analyst' for queries about technical skills, certifications, competencies
- Use 'FINISH' when the task is complete or no agent is needed

DO NOT retrieve data yourself. DO NOT perform deep reasoning.
Simply analyze and route.

Based on the user query and chat history, decide which agent should handle this task."""

# Resume specialist prompt
resume_specialist_prompt = """You are a Resume Specialist Agent.

You handle queries about:
- Work experience and employment history
- Education background
- Career progression
- Job roles and responsibilities

IMPORTANT INSTRUCTIONS:
1. Use the resume_search_tool with appropriate section_filter:
   - For experience queries: section_filter=['EXPERIENCE', 'WORK EXPERIENCE', 'EMPLOYMENT']
   - For education queries: section_filter=['EDUCATION']
   
2. Remember that data is stored at CHUNK LEVEL:
   - Multiple chunks may belong to the same resume
   - Group chunks by resume_id before providing answers
   - Synthesize information across chunks for complete answers
   
3. If a chunk has section='GENERAL':
   - Explicitly inform the user that the context is generic
   - Mention that specific section information may be limited
   
4. Avoid overconfident conclusions if information is incomplete
5. Always provide resume_id in your responses for reference

Provide clear, structured answers based on the retrieved chunks."""

# Skill analyst prompt
skill_analyst_prompt = """You are a Skill Analyst Agent.

You handle queries about:
- Technical skills and competencies
- Certifications and qualifications
- Skill comparisons and gap analysis
- Technology expertise

IMPORTANT INSTRUCTIONS:
1. Use the resume_search_tool with section_filter=['SKILLS', 'TECHNICAL SKILLS'] for skill queries

2. Remember that data is stored at CHUNK LEVEL:
   - Multiple chunks may belong to the same resume
   - Group chunks by resume_id before analyzing skills
   - Synthesize skill information across chunks
   
3. For skill analysis:
   - Extract all relevant skills from retrieved chunks
   - Group by resume_id for comparisons
   - Identify patterns and gaps if requested
   
4. If a chunk has section='GENERAL':
   - Note that skills may be mentioned in general context
   - Look for both explicit and implicit skill mentions
   
5. Provide structured skill analysis with resume_id references

Be thorough and analytical in your skill assessments."""

# Create specialized agents
resume_specialist_agent = create_react_agent(
    model=model,
    tools=[resume_search_tool],
    state_modifier=SystemMessage(content=resume_specialist_prompt)
)

skill_analyst_agent = create_react_agent(
    model=model,
    tools=[resume_search_tool],
    state_modifier=SystemMessage(content=skill_analyst_prompt)
)

# Supervisor routing function
def supervisor_node(state: AgentState):
    messages = state["messages"]
    
    # Create routing prompt
    routing_prompt = f"{supervisor_prompt}\n\nConversation:\n"
    for msg in messages[-5:]:  # Last 5 messages for context
        if isinstance(msg, HumanMessage):
            routing_prompt += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            routing_prompt += f"Assistant: {msg.content}\n"
    
    routing_prompt += "\n\nRoute to: resume_specialist, skill_analyst, or FINISH?"
    
    response = model.invoke([SystemMessage(content=routing_prompt)])
    token_usage.add_usage(response)
    
    content = response.content.lower()
    
    if "resume_specialist" in content or "experience" in content or "education" in content:
        next_agent = "resume_specialist"
    elif "skill_analyst" in content or "skill" in content:
        next_agent = "skill_analyst"
    else:
        next_agent = "FINISH"
    
    return {"next_agent": next_agent}

# Resume specialist node
def resume_specialist_node(state: AgentState):
    result = resume_specialist_agent.invoke(state)
    
    # Track tokens from agent response
    if result.get("messages"):
        last_msg = result["messages"][-1]
        if hasattr(last_msg, 'response_metadata'):
            token_usage.add_usage(last_msg)
    
    return {"messages": result["messages"]}

# Skill analyst node
def skill_analyst_node(state: AgentState):
    result = skill_analyst_agent.invoke(state)
    
    # Track tokens from agent response
    if result.get("messages"):
        last_msg = result["messages"][-1]
        if hasattr(last_msg, 'response_metadata'):
            token_usage.add_usage(last_msg)
    
    return {"messages": result["messages"]}

# Routing logic
def route_after_supervisor(state: AgentState) -> Literal["resume_specialist", "skill_analyst", "__end__"]:
    next_agent = state.get("next_agent", "FINISH")
    
    if next_agent == "resume_specialist":
        return "resume_specialist"
    elif next_agent == "skill_analyst":
        return "skill_analyst"
    else:
        return "__end__"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("resume_specialist", resume_specialist_node)
workflow.add_node("skill_analyst", skill_analyst_node)

# Add edges
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "resume_specialist": "resume_specialist",
        "skill_analyst": "skill_analyst",
        "__end__": END
    }
)
workflow.add_edge("resume_specialist", END)
workflow.add_edge("skill_analyst", END)

# Compile the graph
graph = workflow.compile()

# Main agent interface
def run_agent(user_query: str, chat_history: List = None):
    """
    Main function to run the agent with chat history support.
    
    Args:
        user_query: User's question or query
        chat_history: List of previous messages
    
    Returns:
        Response from the agent and token usage
    """
    token_usage.reset()
    
    messages = []
    
    # Add chat history (last 3 conversations)
    if chat_history:
        messages.extend(chat_history[-6:])  # Last 3 user-assistant pairs
    
    # Add current query
    messages.append(HumanMessage(content=user_query))
    
    # Invoke the graph
    result = graph.invoke(
        {"messages": messages},
        config={"callbacks": [langfuse_handler]}
    )
    
    # Extract final response
    final_message = result["messages"][-1].content
    
    return {
        "response": final_message,
        "token_usage": token_usage.get_usage()
    }