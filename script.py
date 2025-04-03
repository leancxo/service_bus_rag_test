# service_bus_rag_test.py
import redis
import json
import threading
import time
import random
import os
import faiss
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Any
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


###################
# Service Bus
###################
class ServiceBus:
    def __init__(self, host='localhost', port=6379):
        self.redis_client = redis.Redis(host=host, port=port)
        self.pubsub = self.redis_client.pubsub()
        self.subscribers = {}
        self.running = False
        self.listener_thread = None

    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic"""
        self.redis_client.publish(topic, json.dumps(message))

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to a topic with a callback"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
            self.pubsub.subscribe(topic)

        self.subscribers[topic].append(callback)

    def _message_handler(self):
        """Handle incoming messages"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                topic = message['channel'].decode('utf-8')
                data = json.loads(message['data'].decode('utf-8'))

                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        callback(data)

    def start(self):
        """Start the service bus"""
        if not self.running:
            self.running = True
            self.listener_thread = threading.Thread(target=self._message_handler)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            print("Service bus started")

    def stop(self):
        """Stop the service bus"""
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
            self.listener_thread = None
        self.pubsub.unsubscribe()
        print("Service bus stopped")


###################
# Producers
###################
class DataProducer:
    def __init__(self, service_bus, topic, producer_id):
        self.service_bus = service_bus
        self.topic = topic
        self.producer_id = producer_id
        self.running = False

    def generate_data(self) -> Dict[str, Any]:
        """Generate sample data - override in subclasses"""
        raise NotImplementedError

    def start_publishing(self, interval=1.0):
        """Start publishing data at regular intervals"""
        self.running = True
        print(f"Producer {self.producer_id} started publishing to {self.topic}")
        while self.running:
            data = self.generate_data()
            self.service_bus.publish(self.topic, data)
            time.sleep(interval)

    def stop_publishing(self):
        """Stop publishing data"""
        self.running = False
        print(f"Producer {self.producer_id} stopped publishing")


class DocumentProducer(DataProducer):
    """Produces document data for the RAG system"""

    def __init__(self, service_bus, documents):
        super().__init__(service_bus, "documents", "doc_producer")
        self.documents = documents
        self.current_index = 0

    def generate_data(self):
        if not self.documents:
            return {"error": "No documents available"}

        document = self.documents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.documents)

        return {
            "id": f"doc_{random.randint(1000, 9999)}",
            "content": document,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "producer_id": self.producer_id,
                "type": "document"
            }
        }


class QueryProducer(DataProducer):
    """Produces query requests"""

    def __init__(self, service_bus, queries):
        super().__init__(service_bus, "queries", "query_producer")
        self.queries = queries
        self.current_index = 0

    def generate_data(self):
        if not self.queries:
            return {"error": "No queries available"}

        query = self.queries[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.queries)

        return {
            "id": f"query_{random.randint(1000, 9999)}",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "producer_id": self.producer_id,
                "type": "query"
            }
        }


###################
# RAG System
###################
class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Initializing vector store with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        print(f"Vector store initialized with dimension: {self.dimension}")

    def add_document(self, document: Dict):
        """Add a document to the vector store"""
        content = document.get('content', '')
        embedding = self.model.encode([content])[0]

        # Add to FAISS index
        faiss.normalize_L2(np.array([embedding], dtype=np.float32))
        self.index.add(np.array([embedding], dtype=np.float32))

        # Store document
        self.documents.append(document)

        doc_id = len(self.documents) - 1
        print(f"Added document to vector store, ID: {doc_id}, Content: {content[:30]}...")
        return doc_id

    def search(self, query: str, k=5) -> List[Tuple[int, float, Dict]]:
        """Search for documents similar to the query"""
        print(f"Searching for: {query}")
        query_embedding = self.model.encode([query])[0]
        faiss.normalize_L2(np.array([query_embedding], dtype=np.float32))

        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)

        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx != -1 and idx < len(self.documents):
                results.append((idx, float(distance), self.documents[idx]))

        print(f"Found {len(results)} results for query: {query}")
        return results


###################
# Consumers
###################
class RagConsumer:
    def __init__(self, service_bus, vector_store):
        self.service_bus = service_bus
        self.vector_store = vector_store
        self.running = False

    def process_document(self, message: Dict[str, Any]):
        """Process document messages for the RAG system"""
        document_id = self.vector_store.add_document(message)
        print(f"RAG Consumer processed document ID: {document_id}")

    def process_query(self, message: Dict[str, Any]):
        """Process query messages"""
        query = message.get('query', '')
        results = self.vector_store.search(query)

        # Publish results back to the service bus
        response = {
            "query_id": message.get('id'),
            "query": query,
            "results": [
                {
                    "score": float(score),
                    "document": doc.get('content', ''),
                    "document_id": doc.get('id', '')
                }
                for _, score, doc in results
            ],
            "timestamp": datetime.now().isoformat()
        }

        self.service_bus.publish("query_results", response)
        print(f"Published query results for: {query}")

    def start(self):
        """Start consuming messages"""
        self.service_bus.subscribe("documents", self.process_document)
        self.service_bus.subscribe("queries", self.process_query)
        self.running = True
        print("RAG consumer started")

    def stop(self):
        """Stop consuming messages"""
        self.running = False
        print("RAG consumer stopped")


###################
# API Service
###################
class Query(BaseModel):
    text: str
    top_k: int = 5


class QueryResultsSubscriber:
    def __init__(self, service_bus):
        self.service_bus = service_bus
        self.results = {}

    def process_results(self, message: Dict[str, Any]):
        """Store query results"""
        query_id = message.get('query_id')
        if query_id:
            self.results[query_id] = message
            print(f"Received query results for ID: {query_id}")

    def start(self):
        """Start subscribing to query results"""
        self.service_bus.subscribe("query_results", self.process_results)
        print("Query results subscriber started")


class QueryService:
    def __init__(self, service_bus, vector_store):
        self.app = FastAPI(title="RAG Query API")
        self.service_bus = service_bus
        self.vector_store = vector_store
        self.results_subscriber = QueryResultsSubscriber(service_bus)
        self.results_subscriber.start()

        @self.app.post("/query")
        async def query(query: Query):
            # Direct query to vector store
            results = self.vector_store.search(query.text, query.top_k)
            return {
                "query": query.text,
                "results": [
                    {
                        "score": float(score),
                        "document": {
                            "content": doc.get('content', ''),
                            "id": doc.get('id', '')
                        }
                    }
                    for _, score, doc in results
                ]
            }

        @self.app.post("/bus-query")
        async def bus_query(query: Query):
            # Send query through service bus
            query_id = f"query_{random.randint(1000, 9999)}"
            self.service_bus.publish("queries", {
                "id": query_id,
                "query": query.text,
                "timestamp": datetime.now().isoformat()
            })
            return {"message": f"Query sent to service bus, ID: {query_id}"}

        @self.app.get("/query-results/{query_id}")
        async def get_results(query_id: str):
            # Get results from subscriber
            if query_id in self.results_subscriber.results:
                return self.results_subscriber.results[query_id]
            raise HTTPException(status_code=404, detail="Results not found")

    def run(self, host="0.0.0.0", port=8000):
        """Run the API server"""
        print(f"Query service starting on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


###################
# Main Application
###################
def main():
    # Sample documents
    sample_documents = [
        "Service bus architectures allow for decoupled communication between services.",
        "RAG systems combine retrieval with generative AI for more accurate responses.",
        "Redis can be used as a simple message broker for service bus implementations.",
        "Vector databases store embeddings for efficient similarity search.",
        "Microservices can communicate asynchronously through a service bus.",
        "PyCharm is an integrated development environment (IDE) for Python.",
        "FAISS is a library for efficient similarity search developed by Facebook Research.",
        "Sentence transformers convert text into meaningful vector representations.",
        "Event-driven architectures scale well for distributed systems.",
        "Pub/Sub patterns enable loose coupling between publishers and subscribers."
    ]

    # Sample queries
    sample_queries = [
        "How do service buses work?",
        "What is RAG?",
        "How can microservices communicate?",
        "What is FAISS used for?",
        "How do embeddings help with search?"
    ]

    # Initialize components
    print("Initializing service bus...")
    service_bus = ServiceBus()

    print("Initializing vector store...")
    vector_store = VectorStore()

    # Start the service bus
    service_bus.start()

    # Start the RAG consumer
    rag_consumer = RagConsumer(service_bus, vector_store)
    rag_consumer.start()

    # Start the document producer
    doc_producer = DocumentProducer(service_bus, sample_documents)
    doc_thread = threading.Thread(target=doc_producer.start_publishing, args=(2.0,))
    doc_thread.daemon = True
    doc_thread.start()

    # Wait for documents to be indexed
    print("Waiting for documents to be indexed...")
    time.sleep(len(sample_documents) * 2.5)  # Allow time for all documents to be processed

    # Start the query producer
    query_producer = QueryProducer(service_bus, sample_queries)
    query_thread = threading.Thread(target=query_producer.start_publishing, args=(5.0,))
    query_thread.daemon = True
    query_thread.start()

    # Start the query service
    query_service = QueryService(service_bus, vector_store)
    api_thread = threading.Thread(target=query_service.run)
    api_thread.daemon = True
    api_thread.start()

    print("\n" + "=" * 50)
    print("Service Bus with RAG Test Application Running")
    print("=" * 50)
    print("\nAvailable endpoints:")
    print("- POST http://localhost:8000/query - Direct query to vector store")
    print("- POST http://localhost:8000/bus-query - Send query through service bus")
    print("- GET http://localhost:8000/query-results/{query_id} - Get results for a specific query")
    print("\nPress Ctrl+C to stop the application")
    print("=" * 50 + "\n")

    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        doc_producer.stop_publishing()
        query_producer.stop_publishing()
        rag_consumer.stop()
        service_bus.stop()
        print("Application stopped")


if __name__ == "__main__":
    main()