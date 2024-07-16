

import bs4
import os
import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from chainlit.input_widget import Select, Switch, Slider
import chainlit as cl
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from groq import Groq
from langchain_groq import ChatGroq
import time
from dotenv import load_dotenv

load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, streaming=True)

""" embed_model = FastEmbedEmbeddings(model_name ="BAAI/bge-base-en-v1.5")

llm = ChatGroq(
    temperature= 0.2,
    model_name = "Llama3-8B-8k",
    api_key= "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",)    
 """
# PDF as doc
from langchain_community.document_loaders import PyPDFLoader

file_path = "Notes Cindy 5.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# Construct retriever
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":3})

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question
system_prompt = (
    "You are an assistant for question-answering tasks related to Cargowise and air imports."
    "Your name is Tony."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Custom chat profiles
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Air Import Helper 3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://pics.craiyon.com/2023-05-31/84a6e113d2b14f43b4d548a136090532.webp"
        ),
        cl.ChatProfile(
            name="Air Import Helper 4.0",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://pics.craiyon.com/2023-05-31/666002f0de0d44ad9ed480f220db9b1d.webp"
        )
    ]

# Chat settings
@cl.on_chat_start
async def chat_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-4"],
                initial_index=0
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1
            ),
        ]
    ).send()

# Starters
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Summarize air import",
            message="Can you summarize the air import process into brief and well defined steps",
            icon="https://static.vecteezy.com/system/resources/previews/014/237/107/non_2x/intelligent-system-idea-software-development-gradient-icon-vector.jpg"
        ),
        cl.Starter(
            label="Explain a concept",
            message="Can you explain the concept of Incoterms",
            icon="https://www.creativefabrica.com/wp-content/uploads/2022/07/25/Html-File-Line-Gradient-Icon-Graphics-34809782-1-1-580x387.jpg"
        ),
        cl.Starter(
            label="Define a terminology",
            message="Can you define and explain in layman terms \"PTT\"",
            icon="https://png.pngtree.com/png-clipart/20230920/original/pngtree-book-pixel-perfect-gradient-linear-ui-icon-red-color-illustration-vector-png-image_12806820.png"
        ),
        cl.Starter(
            label="Help me use CargoWise",
            message="How do I open an Air Import file on CargoWise?",
            icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfpV1cOOZDFNr2atjHgVNWedyrzJCQ1q2AUw&s"
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    await chat_settings()  # Ensure chat settings are sent on chat start
    cl.user_session.set("session_id", str(uuid.uuid4()))  # Add session ID generation

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(conversational_rag_chain.stream)(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], configurable={"session_id": session_id}),
    ):
        # Ensure chunk is serializable before sending
        if isinstance(chunk, str):
            await msg.stream_token(chunk)
        elif isinstance(chunk, dict):
            await msg.stream_token(chunk.get("answer", ""))
    
    await msg.send()

# Handling settings update
@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

if __name__ == "__main__":
    cl.run() 