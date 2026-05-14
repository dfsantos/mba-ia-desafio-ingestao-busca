from search import search_prompt, PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
system = ("system", PROMPT_TEMPLATE)
chat_prompt = ChatPromptTemplate([system])

pergunta = "Qual o faturamento da empresa X?"

def main():
    chain = search_prompt(pergunta)

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    messages = chat_prompt.format(
        pergunta=pergunta,
        contexto=chain
    )
    result = model.invoke(messages)
    print(result.content)
    

if __name__ == "__main__":
    main()