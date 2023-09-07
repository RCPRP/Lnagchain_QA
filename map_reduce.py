!pip install langchain
!pip install chromadb
!pip install sentence_transformers
!pip install faiss-cpu
%%writefile data.txt
Pregnancy is a sacred time when a mother and her child bond while the child is developing, but this extends far beyond that. The journey from pregnancy to motherhood is something that every new mom wants to cherish.

However, what about soon-to-be mothers in the workforce, who are looking forward to embarking on their journey of motherhood. The support that Indian mothers receive, before and after they have a child, is ingrained in our Indian culture. Therefore, it makes sense to have the same focus on motherhood, even at the workplace. This is possible only with the Maternity Benefit Act sanctioned by the Indian government, which allows soon-to-be mothers to focus on their family, and take some time from work, in the form of maternity leave.

Maternity leave is a work-sanctioned period of absence that working women can utilise before and/or after they deliver a child. The Maternity Benefit Act of 1961 lays down the rules and regulations that concern maternity leave in India. In this Maternity Act, women who are eligible to take a maternity leave and who work at recognized organisations and factories can apply for maternity leave for up to 6 months. Women employees can take their maternity leave either before or after they deliver their child. Their maternity leave can also span a period before and after their delivery as well. During this leave period, the woman's employer is required to pay the woman employee her salary in its entirety.


from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# load document
loader = DirectoryLoader('/content', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

### For multiple documents 
# loaders = [....]
# documents = []
# for loader in loaders:
#     documents.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

'''Convert doc as Embeddings'''
from langchain.embeddings import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)
embeddings = hf

'''Store the embedded version in VectoStore'''
vectorstore = FAISS.from_documents(documents, embeddings) #db = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

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
chain = load_qa_chain(llm=local_llm, chain_type="map_reduce")
query = "What is maternity leave?"
chain.run(input_documents=documents, question=query)
