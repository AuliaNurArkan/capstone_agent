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
