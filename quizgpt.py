import openai
import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from langchain.chat_models import ChatOpenAI

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


api_key = st.sidebar.text_input("Enter your OpenAI API Key")
difficulty = st.sidebar.selectbox("Select Difficulty", ["Easy", "Hard"])
st.sidebar.markdown("[GitHub Repo](https://github.com/magicwizard02/chat_gpt)")

openai.api_key = api_key


function = {
    "name": "create_quiz",
    "description": "Generates a quiz with questions and answers.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(temperature=0.1).bind(
    function_call={"name": "create_quiz"}, functions=[function]
)

prompt = PromptTemplate.from_template(
    "Make a {difficulty} difficulty quiz about the following content: {context}"
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs



with st.sidebar:
    docs = None
    topic = None
    source_choice = st.selectbox("Choose Quiz Source:", ["File", "Wikipedia"])
    
    if source_choice == "File":
        file = st.file_uploader("Upload a .docx, .txt, or .pdf file", type=["pdf", "txt", "docx"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )

else:
    context = format_docs(docs)

    chain = prompt | llm
    response = chain.invoke({"difficulty": difficulty, "context": context})

    response_data = json.loads(response.additional_kwargs["function_call"]["arguments"])

    if "questions" not in st.session_state:
        st.session_state.questions = response_data["questions"]

    incorrect_questions = []

    with st.form("questions_form"):
        st.write(response)  
        user_answers = {}

        for question in st.session_state.questions:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )

            user_answers[question["question"]] = value  

        button = st.form_submit_button() 

    if button:
        incorrect_questions = [
            question for question in st.session_state.questions
            if {"answer": user_answers[question["question"]], "correct": True} not in question["answers"]
        ]

        if not incorrect_questions:
            st.success("You got all answers correct!")
            st.balloons()
            st.session_state.questions = []  
        else:
            st.warning("Some answers were incorrect. Retake it.")
            st.session_state.questions = incorrect_questions 
