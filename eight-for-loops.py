import datetime
# import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser


system_prompt = (
    """
    Extract information from the retrieved context
    {context}

    You must always output only a JSON object with the following keys:
    "Offense information",
    "Vessel name",
    "Vessel flag",
    "Vessel movement",
    "Ownership information",
    "Entity Identity Information",
    "Narrative and Sources",
    "Goods Onboard",
    "Arrest Information",
    "Crew Information and People on Board"
    """
)

human_prompt = (
    """
    Extract the following key information from the text:
    Offense information
    Vessel name
    Vessel flag
    Vessel movement
    Ownership Information
    Entity Identity Information
    Narrative and Sources
    Goods Onboard
    Arrest Information
    Crew Information and People on Board
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def doc_loading_step(article):
    loader = WebBaseLoader(
        web_paths=(article,),
    )
    global docs
    docs = loader.load()


def text_splitting_step(docs, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    global all_splits
    all_splits = text_splitter.split_documents(docs)


def embedding_step(all_splits, embed_model):
    embed_model = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model=embed_model,
    )

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embed_model,
    )

    global retriever
    retriever = vectorstore.as_retriever()


def llm_def_step(llm, temperature, top_k, top_p, verbose):
    global llmodel
    llmodel = ChatOllama(
        base_url='http://localhost:11434',
        model=llm,
        model_kwargs={"response_format": {"type": "json_object"}},
        format="json",
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        verbose=verbose,
    )


def extract_info():
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llmodel
    )

    response = rag_chain.invoke({"input": human_prompt})
    response_json = response.content

    settings = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "embed_model": embed_model, "llm": llmodel.model, "temperature": llmodel.temperature, "top_k": llmodel.top_k, "top_p": llmodel.top_p, "verbose": llmodel.verbose}

    return {"article": article, "settings": settings, "response_metadata": response.response_metadata, "response_json": response_json}


##########################################


articles = [
    "https://www.washingtonpost.com/climate-environment/interactive/2023/map-illegal-fishing",
    "https://www.shetlandtimes.co.uk/2012/07/11/shetland-catch-ordered-to-pay-back-1-5-million-of-black-fish-profits",
    "http://news.bbc.co.uk/2/hi/uk_news/scotland/4442932.stm",
    "https://www.cbc.ca/news/canada/nova-scotia/n-s-boat-captain-fined-fisheries-violations-1.6965198",
    "https://www.nzherald.co.nz/nz/greymouth-fishing-company-westfleet-loses-multi-million-dollar-trawler-for-coral-weighing-less-than-half-a-pound-of-butter/XCCPJFNDRND4XNIXLJ46YLI6B4",
]

c_sizes = [500, 1000, 3000]
c_overlaps = [50, 100, 250]

e_models = [
    "nomic-embed-text",
    "mxbai-embed-large",
    # "snowflake-arctic-embed",
    # "all-minilm",
]

llms = [
    # "aya",
    # "gemma",
    # "llama2",
    "llama3",
    "mistral",
    # "phi3",
    # "qwen2",
]

temps = [0.2, 0.8]
top_ks = [20, 80]
top_ps = [0.2, 0.8]
# verbose_vals = [True, False]


def save_to_file(buffer_list, filename):
    filehandle = open(filename, 'w')
    filebuffer = ["results = [\n\n"] + buffer_list + ["]\n"]
    filehandle.writelines(filebuffer)
    filehandle.close()


out_list = []

buffer_list = []

name_llms = ""
for l in llms:
    name_llms = name_llms + l + "_"

now_string = str(datetime.datetime.now())

filename = name_llms + now_string + ".py"


for article in articles:
    doc_loading_step(article)
    for chunk_size in c_sizes:
        for chunk_overlap in c_overlaps:
            text_splitting_step(docs, chunk_size, chunk_overlap)
            for embed_model in e_models:
                Chroma().delete_collection()
                embedding_step(all_splits, embed_model)
                for llm in llms:
                    for temperature in temps:
                        for top_k in top_ks:
                            for top_p in top_ps:
                                llm_def_step(llm, temperature, top_k, top_p, False)
                                output = extract_info()
                                out_list.append(output)
                                buffer_list.append("\t"+str(output)+",\n\n")
                                save_to_file(buffer_list, filename)
