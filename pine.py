import json

from typing import Any, List, Mapping, Optional
# from mistune import Markdown
# from openai.cli import display
import requests
from llama_index.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.llms.base import llm_completion_callback, CompletionResponseGen
from llama_index import ServiceContext, KnowledgeGraphIndex, StorageContext, VectorStoreIndex, download_loader,set_global_service_context
import os
# import pinecone
from llama_index import VectorStoreIndex, SimpleDirectoryReader
# from llama_index import SimpleCSVReader
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import get_response_synthesizer

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.vector_store import VectorIndexRetriever
# set context window size
context_window = 3500
# set number of output tokens
num_output = 2048

class OurLLM(CustomLLM):

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            # model_name=self.model_url
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Make a POST request to the model's API endpoint with the prompt as data
        model_url = "https://wv7b-satyamkumar-209565-0.datalab.euw1.prod.sgcip.io/v1/completions"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({
            "prompt": "\n\n### Instructions:\n" + prompt + "\n\n### Response:\n",
            "stop": [
                "###"
            ],
            "max_tokens":500
            
        })
        response = requests.request("POST", model_url, headers=headers, data=payload)

        # Parse the response
        response_json = response.json()
        llm_response = response_json["choices"][0]["text"]
        # Print the original length
        

        # Extract the generated text from the response
        # This assumes the response has a field 'generated_text' with the generated text
        # generated_text = response_json['generated_text']
        # Post-process the response to limit its length
        # print(f"Original Response: {llm_response}")
        # max_response_length = 3000  # Set your desired maximum length
        # llm_response = llm_response[:max_response_length]
        # Return a CompletionResponse with the generated text
        return CompletionResponse(text=llm_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Your logic to call your custom LLM using self.model_url
        pass


llm = OurLLM()
service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=152)
set_global_service_context(service_context)
# print("model")
# # api_key = os.environ.get("41329d2c-9102-463f-8aaa-6a1e38a69efb")
# pinecone.init(api_key='41329d2c-9102-463f-8aaa-6a1e38a69efb', environment="gcp-starter")
# print("api")
# pinecone.create_index(
#     name='quickstart', dimension=384,metric="euclidean", pod_type="p1"
# )
# print("create")
# pinecone_index = pinecone.Index("quickstart")
# print(pinecone_index)
# documents = SimpleDirectoryReader("C:\\Users\\HP\\Desktop\\pinecone_lamaindex\\file").load_data()
# # print(documents)
# # SimpleCSVReader = download_loader("SimpleCSVReader")
# # documents = SimpleCSVReader("C:\\Users\\HP\\Desktop\\pinecone_lamaindex\\file\\updated_data.csv").load_data()

# # def upsert_data_to_pinecone(documents):
# #     question_list = []
# #     for i, row in documents.iterrows():
# #         question_list.append(
# #             (
# #                 str(row['id']),
# #                 service_context.encode(row['description']).tolist(),
# #                 {
# #                     'Point_of_contact': row['Point_of_contact'],
# #                     'incident_type': row['incident_type'],
# #                     'Resolution': row['Resolution'],
# #                 }
# #             )
# #         )
# #         if len(question_list) == 50 or len(question_list) == len(documents):
# #             index.upsert(index_name='quickstart',vectors=question_list)
# #             question_list = []
# #     if len(question_list) > 0:
# #      index.upsert(index_name='quickstart',vectors=question_list) 
# # upsert_data_to_pinecone(documents)
# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# print(vector_store)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# print(storage_context)
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context
# )
# index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context
# )
# print(index)

# query_engine = index.as_query_engine()
# print(query_engine)
# response = query_engine.query("Difficulty connecting to the office network or internet what is the resolution ")
# print(response)
# # print(response.get_formatted_sources())
response = llm.complete("tell me about India?")
print(response.text)