from langchain_ollama import OllamaLLM
from langchain_core.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate,)

template = """Bạn là trợ lý ảo của trường đại học Y dược Cần Thơ. Tạo ra duy nhất một đoạn văn khoa học để trả lời câu hỏi sau.
Question: {question}
Answer:"""
prompt_hyde = ChatPromptTemplate.from_template(template)
generate_docs_for_retrieval = (
    prompt_hyde | OllamaLLM(model="llama3.2:1b", temperature=0.3)
)
hyde_doc =  generate_docs_for_retrieval.invoke({"question": "bạn có thông tin gì"})

# llm = OllamaLLM(model="llama3.1")
# response = llm.invoke("xin chào")
print(hyde_doc)