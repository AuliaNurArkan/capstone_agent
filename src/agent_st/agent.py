from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Literal

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

class AgentInput(BaseModel):
    """Input for search resume."""
    query: str = Field(description="Search query from user")
    history: str = Field(description="summary of chat history")

@tool
def search_resume(query: str, k: int = 5) -> list[str]:
    """Retrieve relevant resumes on the query."""


    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )

    docs = vector_store.similarity_search_with_score(query, k=k)
    if docs:
        formatted_results = []
        for idx, result in enumerate(docs):
            formatted_results.append(f"""
Resume {idx + 1}:
- ID: {result[0].metadata.get('row_index', 'N/A')}
- Category: {result[0].metadata.get('category', 'N/A')}
- Relevance Score: {result[1]:.3f}
- Content Preview: {result[0].page_content[:300]}...
""")
        
        context = "\n".join(formatted_results)
        return context
    return "No relevant documents found."

@tool
def search_resume_skill(query: str, k: int = 5) -> list[str]:
    """Retrieve relevant resumes on the query."""


    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )

    docs = vector_store.similarity_search_with_score(query, k=k)
    if docs:
        formatted_data = []
        for idx, result in enumerate(docs):
            formatted_data.append(f"""
Resume {idx + 1} ({result[0].metadata.get('row_index', 'N/A')}):
{result[0].page_content[:500]}...
""")
        
        context = "\n".join(formatted_data)
        return context
    return "No relevant documents found."

lf_resume_search = lf.get_prompt("resume_search_agent").get_langchain_prompt()
import os
from typing import Annotated, List, Union, TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.callbacks import get_openai_callback
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

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
    temperature=0
)

# --- TOOLS DEFINITION (SECTION-AWARE) ---

@tool
def resume_search_tool(query: str, section_filter: List[str] = None):
    """
    Search the resume database deeply. 
    Use 'section_filter' to narrow down results to: ['EXPERIENCE', 'SKILLS', 'EDUCATION', 'SUMMARY'].
    Results are returned at the chunk level with full metadata.
    """
    qdrant_filter = None
    if section_filter:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.section",
                    match=models.MatchAny(any=section_filter)
                )
            ]
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )

    # Retrieval: Fetching 6 most relevant chunks
    docs = vector_store.similarity_search(query, k=6, filter=qdrant_filter)
    
    formatted_results = []
    for doc in docs:
        res_id = doc.metadata.get("resume_id", "N/A")
        sec = doc.metadata.get("section", "GENERAL")
        # Prepending labels so the LLM understands the context of the fragmented text
        content = f"[RESUME_ID: {res_id} | SECTION: {sec}]\n{doc.page_content}"
        formatted_results.append(content)
    
    return "\n\n---\n\n".join(formatted_results) if formatted_results else "No relevant data found."

# --- AGENT STATE & SUB-AGENTS ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "Conversation history"]
    next_agent: str

# 1. Specialist: Work Experience & Education
resume_spec_prompt = (
    "You are an HR Expert specializing in work history and education analysis. "
    "Your task is to analyze candidate experiences in detail. "
    "You MUST use 'resume_search_tool' with filters ['EXPERIENCE', 'EDUCATION', 'SUMMARY']. "
    "CRITICAL: Since data is stored in chunks, group and synthesize information by the same RESUME_ID "
    "to provide a coherent narrative of a candidate's background."
)
resume_specialist_agent = create_react_agent(
    model, tools=[resume_search_tool], state_modifier=resume_spec_prompt
)

# 2. Specialist: Skill Analysis
skill_analyst_prompt = (
    "You are a Technical Recruiter specializing in skill assessments and certifications. "
    "Your task is to identify technical competencies or perform skill gap analysis. "
    "You MUST use 'resume_search_tool' with filters ['SKILLS']. "
    "Compare skills across candidates if requested. Always link skills to their respective RESUME_ID."
)
skill_analyst_agent = create_react_agent(
    model, tools=[resume_search_tool], state_modifier=skill_analyst_prompt
)

# --- SUPERVISOR LOGIC ---

class SupervisorOutput(TypedDict):
    next_agent: Union[Literal["resume_specialist", "skill_analyst", "FINISH"]]

def supervisor_node(state: AgentState):
    """Decides which specialized agent should handle the user request."""
    system_prompt = (
        "You are the HR Agent Supervisor. Your job is to delegate tasks based on user intent: \n"
        "- If the query is about work experience, career history, or education -> route to 'resume_specialist'.\n"
        "- If the query is about technical skills, tools, or specific competencies -> route to 'skill_analyst'.\n"
        "- If you have gathered enough information to provide a comprehensive final answer -> choose 'FINISH'."
    )
    messages = state["messages"]
    response = model.with_structured_output(SupervisorOutput).invoke(
        [{"role": "system", "content": system_prompt}] + messages,
        config={"callbacks": [langfuse_handler]}
    )
    return {"next_agent": response["next_agent"]}

# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("resume_specialist", lambda state: resume_specialist_agent.invoke(state, config={"callbacks": [langfuse_handler]}))
workflow.add_node("skill_analyst", lambda state: skill_analyst_agent.invoke(state, config={"callbacks": [langfuse_handler]}))

workflow.set_entry_point("supervisor")

def router(state):
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]

workflow.add_conditional_edges("supervisor", router)
workflow.add_edge("resume_specialist", "supervisor")
workflow.add_edge("skill_analyst", "supervisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- HELPER FUNCTION FOR STREAMLIT + TOKEN TRACKING ---

def ask_agent_with_stats(query: str, thread_id: str):
    """
    Main function for Streamlit UI. Returns final answer and usage statistics.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler]
    }
    inputs = {"messages": [HumanMessage(content=query)]}
    
    with get_openai_callback() as cb:
        result = app.invoke(inputs, config=config)
        final_message = result["messages"][-1].content
        
        stats = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost": cb.total_cost
        }
    
    return final_message, stats
resume_search_agent = create_agent(
    model=model,
    tools=[search_resume],
    system_prompt=lf_resume_search
)

lf_skill_analyze = lf.get_prompt("skill_analyze_agent").get_langchain_prompt()

skill_analyze_agent = create_agent(
    model=model,
    tools=[search_resume_skill],
    system_prompt=lf_skill_analyze
)

@tool(
        args_schema=AgentInput
)
def resume_search(query: str, history: str) -> str:
    """Tool to search resumes using the resume search agent.
    Use this when the user wants to  find/search for specific candidates or resumes

    query: "find HR managers", "search for candidates with X skill".
    history: chat history summary
    """
    result = resume_search_agent.invoke({
        "messages": [{"role": "user", "content": query + "history chat: " + history}]
    }, config={"callbacks": [langfuse_handler]})
    return result["messages"][-1].text

@tool(
        args_schema=AgentInput
)
def skill_analyze(query: str, history: str) -> str:
    """Tool to Analyze skills, create comparisons, identify gaps
    Use when: User asks about skills, wants analysis or comparisons
    
    query: "what skills does", "compare skills", "skills gap analysis
    history: chat history
    """
    result = skill_analyze_agent.invoke({
        "messages": [{"role": "user", "content": query + "history chat: " + history}]
    }, config={"callbacks": [langfuse_handler]})
    return result["messages"][-1].text

lf_supervisor = lf.get_prompt("supervisor_agent").get_langchain_prompt()
# supervisor
supervisor_agent = create_agent(
    model=model,
    tools=[search_resume, search_resume_skill],
    system_prompt=lf_supervisor
)
