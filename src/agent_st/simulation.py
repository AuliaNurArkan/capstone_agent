
import streamlit as st
from langchain_core.messages import ToolMessage
from agent_st.agent import supervisor_agent as agent
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

def send_chat(question: str, history: str) -> dict:
    result = agent.invoke(
        {"messages": history + [{"role": "user", "content": question}]},
        config={"callbacks": [langfuse_handler]}
    )
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            # Fallback for older or different structures
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_message_content = message.content
            tool_messages.append(tool_message_content)

    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
    return response

st.title("Chatbot HR ")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about Resumes!"):
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("AI"):
        response = send_chat(prompt, messages_history)
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("**Tool Calls:**"):
        st.code(response["tool_messages"])

    with st.expander("**History Chat:**"):
        st.code(history)

    with st.expander("**Usage Details:**"):
        st.code(f'input token : {response["total_input_tokens"]}\noutput token : {response["total_output_tokens"]}')