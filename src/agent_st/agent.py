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
import re

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


def summarize_history(full_history: str, max_length: int = 200) -> str:
    """
    Summarize chat history untuk mengurangi token usage.
    
    Args:
        full_history: Full chat history string
        max_length: Maximum character length for summary
    
    Returns:
        Summarized history in structured format
    """
    if not full_history or len(full_history.strip()) == 0:
        return ""
    
    if len(full_history) <= max_length:
        return full_history
    
    # Extract key information using patterns
    summary_parts = []
    
    # 1. Extract search queries/intents
    search_patterns = [
        r"(?:find|search|looking for|need)\s+(.{10,50})",
        r"(?:candidates?|people)\s+(?:with|who have)\s+(.{10,50})"
    ]
    searches = []
    for pattern in search_patterns:
        matches = re.findall(pattern, full_history.lower(), re.IGNORECASE)
        searches.extend([m.strip() for m in matches[:2]])  # Max 2
    
    if searches:
        summary_parts.append(f"Looking for: {', '.join(searches[:2])}")
    
    # 2. Extract candidate IDs mentioned
    id_pattern = r"(?:ID|id|candidate)\s*:?\s*(\d+)"
    ids = re.findall(id_pattern, full_history)
    unique_ids = list(set(ids))[:3]  # Max 3 IDs
    
    if unique_ids:
        summary_parts.append(f"Discussed IDs: {', '.join(unique_ids)}")
    
    # 3. Extract roles/categories mentioned
    role_pattern = r"(?:data scientist|engineer|developer|manager|analyst|designer)s?"
    roles = re.findall(role_pattern, full_history.lower(), re.IGNORECASE)
    unique_roles = list(set(roles))[:2]  # Max 2 roles
    
    if unique_roles:
        summary_parts.append(f"Roles: {', '.join(unique_roles)}")
    
    # 4. Extract key findings
    findings_patterns = [
        r"found\s+(\d+)\s+(?:candidate|result)",
        r"top\s+(?:match|candidate).*?(id\s*:?\s*\d+)"
    ]
    findings = []
    for pattern in findings_patterns:
        matches = re.findall(pattern, full_history.lower(), re.IGNORECASE)
        findings.extend(matches[:1])
    
    if findings:
        summary_parts.append(f"Previous findings: {' '.join(str(f) for f in findings)}")
    
    # Build summary
    if summary_parts:
        summary = "HISTORY SUMMARY:\n- " + "\n- ".join(summary_parts)
    else:
        # Fallback: take last N characters
        summary = "Recent context: " + full_history[-max_length:]
    
    # Final check: ensure within max_length
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary

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
- ID: {result[0].metadata.get('resume_id', 'N/A')}
- Category: {result[0].metadata.get('category', 'N/A')}
- Section: {result[0].metadata.get('section_type', 'N/A')}
- Relevance Score: {result[1]:.3f}
- Content Preview: {result[0].page_content[:300]}...
""")
        
        context = "\n".join(formatted_results)
        return context
    return "No relevant documents found."

@tool
def search_resume_skill(query: str, k: int = 5) -> list[str]:
    """Retrieve relevant resumes based on skills query."""

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
Resume {idx + 1} (ID: {result[0].metadata.get('resume_id', 'N/A')}):
Section: {result[0].metadata.get('section_type', 'N/A')}
Category: {result[0].metadata.get('category', 'N/A')}
{result[0].page_content[:500]}...
""")
        
        context = "\n".join(formatted_data)
        return context
    return "No relevant documents found."

@tool
def search_resume_section(query: str, section_filter: str = None, k: int = 5) -> list[str]:
    """Retrieve specific sections from resumes (e.g., experience, education, projects).
    
    Args:
        query: Search query
        section_filter: Filter by section type (skills, experience, education, projects, certifications, summary)
        k: Number of results to return
    """

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )

    docs = vector_store.similarity_search_with_score(query, k=k*2 if section_filter else k)
    

    if section_filter and docs:
        filtered_docs = []
        for doc, score in docs:
            if doc.metadata.get('section_type', '').lower() == section_filter.lower():
                filtered_docs.append((doc, score))
                if len(filtered_docs) >= k:
                    break
        docs = filtered_docs
    
    if docs:
        formatted_data = []
        for idx, result in enumerate(docs):
            formatted_data.append(f"""
Resume {idx + 1}:
- ID: {result[0].metadata.get('resume_id', 'N/A')}
- Category: {result[0].metadata.get('category', 'N/A')}
- Section Type: {result[0].metadata.get('section_type', 'N/A').upper()}
- Relevance Score: {result[1]:.3f}
- Content:
{result[0].page_content}
---
""")
        
        context = "\n".join(formatted_data)
        return context
    return f"No relevant {section_filter or 'resume'} sections found."

@tool
def verify_resume_info(query: str, k: int = 3) -> list[str]:
    """Verify or validate information about candidates mentioned in conversation.
    
    IMPORTANT: This tool returns RAW DATA for analysis.
    DO NOT show all results to user. Extract only the answer to their specific question.
    
    Args:
        query: Verification query with candidate context
        k: Number of results to retrieve for verification
    """

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
Verification Result {idx + 1}:
- Resume ID: {result[0].metadata.get('resume_id', 'N/A')}
- Category: {result[0].metadata.get('category', 'N/A')}
- Section: {result[0].metadata.get('section_type', 'N/A')}
- Score: {result[1]:.3f}
- Evidence:
{result[0].page_content}
---
""")
            
        context = "\n".join(formatted_data)
        # ✅ TAMBAHKAN INSTRUKSI DI OUTPUT
        return f"""[ANALYSIS DATA - DO NOT SHOW ALL TO USER]
        {context}
        [INSTRUCTION TO AGENT: Read the above data and answer ONLY the user's specific question. Be concise (2-3 sentences). DO NOT list all verification results.]"""
    return "No information found to verify the statement."


# Agent 1: Resume Search Agent
lf_resume_search = lf.get_prompt("resume_search_agent").get_langchain_prompt()

resume_search_agent = create_agent(
    model=model,
    tools=[search_resume],
    system_prompt=lf_resume_search
)

# Agent 2: Skill Analyze Agent
lf_skill_analyze = lf.get_prompt("skill_analyze_agent").get_langchain_prompt()

skill_analyze_agent = create_agent(
    model=model,
    tools=[search_resume_skill],
    system_prompt=lf_skill_analyze
)

# Agent 3: Section Search Agent
lf_section_search = lf.get_prompt("section_search_agent").get_langchain_prompt()

section_search_agent = create_agent(
    model=model,
    tools=[search_resume_section],
    system_prompt=lf_section_search
)

# Agent 4: Verification Agent
lf_verification = lf.get_prompt("verification_agent").get_langchain_prompt()

verification_agent = create_agent(
    model=model,
    tools=[verify_resume_info],
    system_prompt=lf_verification
)

@tool(args_schema=AgentInput)
def resume_search(query: str, history: str) -> str:
    """Tool to search resumes using the resume search agent.
    Use this when the user wants to find/search for specific candidates or resumes

    query: "find HR managers", "search for candidates with X skill".
    history: chat history summary
    """
    # Summarize history if too long
    history_summary = summarize_history(history, max_length=200)
    
    result = resume_search_agent.invoke({
        "messages": [{"role": "user", "content": f"{query}\n\n{history_summary}"}]
    }, config={"callbacks": [langfuse_handler]})
    return result["messages"][-1].content

@tool(args_schema=AgentInput)
def skill_analyze(query: str, history: str) -> str:
    """Tool to Analyze skills, create comparisons, identify gaps
    Use when: User asks about skills, wants analysis or comparisons
    
    query: "what skills does", "compare skills", "skills gap analysis"
    history: chat history
    """
    # Summarize history if too long
    history_summary = summarize_history(history, max_length=200)
    
    result = skill_analyze_agent.invoke({
        "messages": [{"role": "user", "content": f"{query}\n\n{history_summary}"}]
    }, config={"callbacks": [langfuse_handler]})
    return result["messages"][-1].content

@tool(args_schema=AgentInput)
def section_search(query: str, history: str) -> str:
    """Tool to search specific sections in resumes (experience, education, projects, etc.)
    Use when: User asks about specific resume sections or detailed information from particular sections
    
    query: "show me experience sections", "find education background", "what projects have candidates done"
    history: chat history
    """
    # Summarize history if too long
    history_summary = summarize_history(history, max_length=200)
    
    result = section_search_agent.invoke({
        "messages": [{"role": "user", "content": f"{query}\n\n{history_summary}"}]
    }, config={"callbacks": [langfuse_handler]})
    return result["messages"][-1].content

@tool(args_schema=AgentInput)
def verify_statement(query: str, history: str) -> str:
    """Tool to verify or validate claims/statements about candidates
    Use when: User asks verification questions like "is it true...", "does candidate X have...", "verify that..."
    
    query: "is this candidate 5 years experience?", "does candidate have AWS cert?", "verify their education"
    history: chat history with candidate context
    """
    # Extract candidate IDs dari history
    history_summary = summarize_history(history, max_length=200)
    
    # Extract IDs yang disebutkan di history
    candidate_ids = re.findall(r'(?:ID|id|Resume)\s*:?\s*(\d+)', history)
    
    # Build enhanced query
    enhanced_query = query
    if candidate_ids:
        unique_ids = list(set(candidate_ids))[:3]  # Max 3 IDs
        enhanced_query = f"{query}\n\nContext: Asking about Resume IDs: {', '.join(unique_ids)}\n\n{history_summary}"
    else:
        enhanced_query = f"{query}\n\n{history_summary}"
    
    # Instruksi eksplisit ke nested agent
    enhanced_query += "\n\nIMPORTANT: Provide a SHORT, DIRECT answer (2-3 sentences). DO NOT list all verification results."
    
    result = verification_agent.invoke({
        "messages": [{"role": "user", "content": enhanced_query}]
    }, config={"callbacks": [langfuse_handler]})
    
    return result["messages"][-1].content

# Supervisor Agent
lf_supervisor = lf.get_prompt("supervisor_agent").get_langchain_prompt()

supervisor_agent = create_agent(
    model=model,
    tools=[resume_search, skill_analyze, section_search, verify_statement],
    system_prompt=lf_supervisor
)