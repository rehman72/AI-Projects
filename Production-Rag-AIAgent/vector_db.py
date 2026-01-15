from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams,Distance,PointStruct


class QdrantStorage:
    def __init__(self,collection="docs",dim=768):
        self.client=QdrantClient(host="127.0.0.1",port=6333,timeout=30)
        self.collection=collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim,distance=Distance.COSINE)
                )

    def upsert(self,ids,vector,payloads):
        points=[PointStruct(id=ids[i],vector= vector[i],payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection,points=points)
    

    def search(self,query_vector,top_k:int=5):
        results=self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )

        context=[]
        sources=set()


        for r in results.points:
            payload=getattr(r,"payload",None) or {}
            text=payload.get("text","")
            source=payload.get("source","")
            
            if text:
                context.append(text)
                sources.add(source)

        return {"contexts":context,"sources": list(sources)}

            



