import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START,END, StateGraph
from langchain_groq import ChatGroq
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# Building a retriver function 
def build_retriever(pdf_path: str, collection_name: str, persist_directory: str):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )

    # If Chroma DB already exists, load it
    if os.path.exists(persist_directory):
        print(f"Loading existing vector store: {collection_name}")

        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    else:
        print(f"Creating vector store: {collection_name}")

        reader = PdfReader(pdf_path)

        docs = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num
                    }
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        print("Chunking done")

        vector_store = Chroma.from_documents(
            collection_name=collection_name,
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        print("Vector store created")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
academic_retriever = build_retriever(
    "academics_handbook.pdf",
    "academic_db",
    "./chroma_academic"
)

fees_retriever = build_retriever(
    "fees_structure.pdf",
    "fee_db",
    "./chroma_fee"
)

llm= ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

#Step 2: Create a state 
class State(TypedDict):
    program: str
    messages:Annotated[list, add_messages]
    query_type: str
    retrieved_context:str

#Step 3: Nodes generation

## Classifier Node which will classify the user query into one of the three categories: academic, fee, or general.
def classifier_node(state:State)->dict:
    """Look at the latest user messages and decide which path to take. """
    last_message= state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
        )

    
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    
    return {"query_type" : category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academic handbook."""
    query = state["messages"][-1].content

    history = "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in state["messages"][:-1]
    )

    retrieval_query = f"""
        Previous conversation:
        {history}

        Current question:
        {query}
        """

    docs = academic_retriever.invoke(retrieval_query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content

    history = "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in state["messages"][:-1]
    )

    retrieval_query = f"""
        Previous conversation:
        {history}

        Current question:
        {query}
        """

    docs = fees_retriever.invoke(retrieval_query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": ""}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's program."""
    query = state["messages"][-1].content
    program = state["program"]
    context = state["retrieved_context"]
    history = "\n".join(
    f"{msg.type}: {msg.content}"
    for msg in state["messages"][:-1]
)

    if not context:
        prompt = (
        f"You are a friendly college assistant talking to a {program} student.\n\n"
        f"Previous conversation:\n{history}\n\n"
        f"Current question:\n{query}\n\n"
        f"Answer naturally and clearly."
        )
    else:
        prompt = (
            f"You are a college assistant helping a {program} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programs, highlight the one relevant to {program} if possible.\n\n"
            f"Previous conversation:\n{history}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}

#Step 4 - router function 

def route_query(state:State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


#Step 5 - Create the state graph
graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

graph.add_edge(START,"classifier")
graph.add_conditional_edges("classifier",route_query)
graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")
graph.add_edge("response",END)

app= graph.compile()

#Step 6 : Run the code
print("Welcome to the college assistant")
print("-"*55)
print("Choose your program: \n")
print("1.BCA")
print("2.BBA")
print("3.BCom")
choice= input("\n Enter only digits 1,2 or 3:")
program_map= {
    "1":"BCA",
    "2":"BBA",
    "3":"BCom"
}

student_program= program_map.get(choice,"BCA")
print(f"\nGreat to know you are a {student_program} student.")

conversation = []
while True:

    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        break

    conversation.append(
        ("human", user_query)
    )

    result = app.invoke({
        "program": student_program,
        "messages": conversation
    })

    conversation = result["messages"]

    print(
        f"Assistant: {result['messages'][-1].content}"
    )
