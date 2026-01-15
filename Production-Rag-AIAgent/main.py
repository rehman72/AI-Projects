from google import genai
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import pdb
import os
import datetime
from data_loader import load_and_chunk_pdf,embed_text
from vector_db import QdrantStorage
from custome_types import RAQueryResult,RAGUpsertResult,RAGSearchResult,RAGChunkAndSrc
load_dotenv()

client=genai.Client()

inject_client=inject_client=inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer= inngest.PydanticSerializer()
)
# inject Function

@inject_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx:inngest.Context)->RAGChunkAndSrc:
        pdf_path=ctx.event.data["pdf_path"]
        source_id=ctx.event.data.get("source_id",pdf_path)
        chunks=load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks,source_id=source_id)

    def _upsert(chunk_and_src:RAGChunkAndSrc)->RAGUpsertResult:
        chunks=chunk_and_src.chunks
        source_id=chunk_and_src.source_id
        vecs=embed_text(chunks)
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,f"{source_id}: {i}")) for i in range (len(chunks))]

        payloads=[{"source": source_id,"text": chunks[i]} for i in range (len(chunks))]
        QdrantStorage().upsert(ids,vecs,payloads)

        return RAGUpsertResult(ingested=len(chunks))
    chunks_and_src=await ctx.step.run("load-and-chunk",lambda:_load(ctx),output_type=RAGChunkAndSrc)
    inngested=await ctx.step.run("embed-and-upsert",lambda:_upsert(chunks_and_src),output_type=RAGUpsertResult)
    return inngested.model_dump()
@inject_client.create_function(
        fn_id="RAG: Query PDF",
        trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx:inngest.Context):
    def _search(question,top_k:int=5):
        query_vec=embed_text([question])[0]
        store=QdrantStorage()
        pdb.set_trace()
        found=store.search(query_vec,top_k)
        return RAGSearchResult(contexts=found["contexts"],sources=found["sources"])
    question=ctx.event.data["question"]
    top_k=ctx.event.data.get("top_k",5)
    found=await ctx.step.run("embed-and-search",lambda:_search(question,top_k),output_type=RAGSearchResult)
    context_block="\n\n".join(f"- {c}" for c in found.contexts)
    user_content=(
        "Use the following context to answer the question.\n\n"
        f"Context:\n {context_block}.\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above"
    )
    
    response=client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [{"text": user_content}]
            }
        ],
        config={
            "temperature":0.7,
                "max_output_tokens":1024
        }
        
    )
    return {
        "answer": response.text,
        "sources": found.sources
    }
    
    



    
    

app=FastAPI()

inngest.fast_api.serve(
    app,
    inject_client, 
    [rag_ingest_pdf,rag_query_pdf_ai]
)






