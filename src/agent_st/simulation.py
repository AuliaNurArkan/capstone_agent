import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from agent_st.agent import supervisor_agent as agent
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

def send_chat(question: str, history: list) -> dict:
    # 1. Gabungkan history dengan input baru
    input_messages = history + [{"role": "user", "content": question}]
    
    # 2. Panggil Agent
    result = agent.invoke(
        {"messages": input_messages},
        config={"callbacks": [langfuse_handler]}
    )
    
    # 3. Ambil jawaban terakhir (AI Answer)
    answer = result["messages"][-1].content

    # 4. Filter hanya pesan yang muncul SETELAH input user terakhir (Turn Sekarang)
    len_input = len(input_messages)
    new_messages = result["messages"][len_input:]

    # 5. Hitung Token Usage
    total_input_tokens = 0
    total_output_tokens = 0
    for message in result["messages"]:
        meta = message.response_metadata
        if "usage_metadata" in meta:
            total_input_tokens += meta["usage_metadata"].get("input_tokens", 0)
            total_output_tokens += meta["usage_metadata"].get("output_tokens", 0)
        elif "token_usage" in meta:
            total_input_tokens += meta["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += meta["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    # 6. EKSTRAK TOOL CALLS (Hanya Nama Tool dan Argumennya)
    actual_tool_calls = []
    for message in new_messages:
        # Cek apakah pesan dari AI mengandung instruksi panggil tool
        if isinstance(message, AIMessage) and message.tool_calls:
            for tc in message.tool_calls:
                actual_tool_calls.append({
                    "tool": tc["name"],
                    "args": tc["args"]
                })

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_calls": actual_tool_calls # Ini yang akan ditampilkan di UI
    }

# --- BAGIAN UI STREAMLIT ---

st.title("Chatbot HR Specialist")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history chat di layar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya seputar resume..."):
    # Simpan history user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Kirim ke Agent (Kirim history maksimal 10 pesan terakhir)
    history_to_send = st.session_state.messages[:-1][-10:]
    
    with st.chat_message("assistant"):
        response = send_chat(prompt, history_to_send)
        st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    with st.expander("**Tool Calls:**"):
        if response["tool_calls"]:
            # Tampilan ini akan mirip dengan yang diminta di ketentuan (JSON metadata)
            st.json(response["tool_calls"])
        else:
            st.info("No tools called (Answered from context/history).")

    with st.expander("**History Chat (Raw):**"):
        st.write(history_to_send)

    with st.expander("**Usage Details:**"):
        st.write(f"Input Tokens: {response['total_input_tokens']}")
        st.write(f"Output Tokens: {response['total_output_tokens']}")
        st.write(f"Cost: Rp {response['price']:.2f}")