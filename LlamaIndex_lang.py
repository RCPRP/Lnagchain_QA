from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, PromptHelper, LLMPredictor
from llama_index import LangchainEmbedding, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



####embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

from langchain.embeddings import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))

'''Create your local llm'''
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
model_id = 'google/flan-t5-base'# going for a smaller model as we dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)
local_llm = HuggingFacePipeline(pipeline=pipe)

# set maximum input size
max_input_size = 2048
# set number of output tokens
num_outputs = 512
# set maximum chunk overlap
max_chunk_overlap = 20.0
# set chunk size limit
chunk_size_limit = 300
prompt_helper = PromptHelper(max_input_size, num_outputs)

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=LLMPredictor(local_llm), prompt_helper=prompt_helper, chunk_size_limit=chunk_size_limit)
# build index
documents = SimpleDirectoryReader('/content').load_data()
# loader = DirectoryLoader('/content', glob="./*.txt", loader_cls=TextLoader)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(documents)

new_index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = new_index.as_query_engine(
    response_mode='no_text',
    verbose=True,
    similarity_top_k=2
)

template = """
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
### Human: Given the context:
---
{context}
---
Answer the following question:
---
{input}
### Assistant:
"""
from langchain import LLMChain, PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=template,
)

chain = LLMChain(
    llm=local_llm,
    prompt=prompt,
    verbose=True
)
# chain = RetrievalQA.from_llm(llm=local_llm,return_source_documents=True,prompt=prompt,verbose=True)

user_input= "what is maternity leave?"
context = query_engine.query(user_input)
concatenated_context = ' '.join(map(str,  [node.node.text for node in context.source_nodes]))
response = chain.run({"context": concatenated_context, "input": user_input})
response
