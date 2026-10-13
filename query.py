import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

prompt = ChatPromptTemplate.from_template("""
基于以下文献内容回答问题。如果文献中没有相关信息，就说"文献中未找到"。

文献：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

questions = [
    "肺泡干细胞如何参与肺损伤后的修复？",
    "肺基底干细胞在流感病毒感染后如何增殖和分化？",
    "干细胞转分化在细支气管再生中的作用是什么？",
    "How do alveolar stem cells contribute to lung repair after injury?",
    "迷走神经如何调控肺部干细胞的修复？"
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"问题: {q}")
    print('='*50)
    answer = chain.invoke(q)
    print(answer)