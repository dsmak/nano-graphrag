"""
PostgreSQL-based storage backends for nano-graphrag

This module provides PostgreSQL implementations of the base storage classes:
- PostgreSQLKVStorage: Key-value storage using graphrag_kv_storage table
- PostgreSQLVectorStorage: Vector storage using graphrag_vector_storage table  
- PostgreSQLGraphStorage: Graph storage using graphrag_graph_nodes/edges tables

All storage classes use the 'namespace' field for multi-tenant isolation (user_id).
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Union, Optional, Tuple
import numpy as np
import asyncpg

from nano_graphrag.base import (
    BaseKVStorage,
    BaseVectorStorage, 
    BaseGraphStorage,
)
from nano_graphrag._utils import EmbeddingFunc

logger = logging.getLogger(__name__)


class PostgreSQLKVStorage(BaseKVStorage[Dict[str, Any]]):
    """PostgreSQL-based key-value storage backend"""
    
    def __init__(self, namespace: str, global_config: dict):
        super().__init__()
        self.namespace = namespace
        self.global_config = global_config
        
        # Import db_manager from the database module
        try:
            from database import db_manager
            self.db_pool = db_manager.pool
        except ImportError:
            raise RuntimeError("Database manager not available. Cannot initialize PostgreSQL storage.")
        
        logger.info(f"PostgreSQL KV Storage initialized for namespace: {namespace}")
    
    async def all_keys(self) -> List[str]:
        """Get all keys in this namespace"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key FROM graphrag_kv_storage WHERE namespace = $1",
                self.namespace
            )
            return [row['key'] for row in rows]
    
    async def get_by_id(self, key: str) -> Union[Dict[str, Any], None]:
        """Get value by key"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM graphrag_kv_storage WHERE namespace = $1 AND key = $2",
                self.namespace, key
            )
            if row:
                return dict(row['value'])
            return None
    
    async def get_by_ids(self, keys: List[str]) -> List[Union[Dict[str, Any], None]]:
        """Get multiple values by keys"""
        if not keys:
            return []
        
        async with self.db_pool.acquire() as conn:
            # Use array comparison for efficient bulk fetch
            rows = await conn.fetch(
                "SELECT key, value FROM graphrag_kv_storage WHERE namespace = $1 AND key = ANY($2)",
                self.namespace, keys
            )
            
            # Create lookup dict and preserve order
            results_dict = {row['key']: dict(row['value']) for row in rows}
            return [results_dict.get(key) for key in keys]
    
    async def upsert(self, key: str, value: Dict[str, Any]) -> None:
        """Insert or update a key-value pair"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graphrag_kv_storage (namespace, key, value)
                VALUES ($1, $2, $3)
                ON CONFLICT (namespace, key) 
                DO UPDATE SET value = $3, updated_at = NOW()
                """,
                self.namespace, key, json.dumps(value)
            )
    
    async def drop(self) -> None:
        """Clear all data for this namespace"""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM graphrag_kv_storage WHERE namespace = $1",
                self.namespace
            )
            deleted_count = int(result.split()[-1])
            logger.info(f"Dropped {deleted_count} KV entries for namespace {self.namespace}")

    async def index_start_callback(self):
        """Called when indexing starts - no-op for PostgreSQL"""
        pass


class PostgreSQLVectorStorage(BaseVectorStorage):
    """PostgreSQL-based vector storage backend"""
    
    def __init__(self, namespace: str, global_config: dict, embedding_func: EmbeddingFunc, **kwargs):
        super().__init__()
        self.namespace = namespace
        self.global_config = global_config
        self.embedding_func = embedding_func
        self.meta_fields = set()
        
        # Import db_manager from the database module
        try:
            from database import db_manager
            self.db_pool = db_manager.pool
        except ImportError:
            raise RuntimeError("Database manager not available. Cannot initialize PostgreSQL storage.")
        
        logger.info(f"PostgreSQL Vector Storage initialized for namespace: {namespace}")
    
    async def upsert(self, vector_id: str, data: Dict[str, Any]) -> None:
        """Insert or update a vector with metadata"""
        # Extract content for embedding
        content = data.get('content', '')
        if not content:
            logger.warning(f"No content found for vector {vector_id}")
            return
            
        # Generate embedding
        try:
            embedding = await self.embedding_func([content])
            if not embedding or len(embedding) == 0:
                logger.error(f"Failed to generate embedding for vector {vector_id}")
                return
            embedding_array = embedding[0]  # Get first (and only) embedding
        except Exception as e:
            logger.error(f"Embedding generation failed for {vector_id}: {e}")
            return
            
        # Prepare metadata (exclude content to avoid duplication)
        metadata = {k: v for k, v in data.items() if k != 'content'}
        metadata['content'] = content  # Keep content in metadata for retrieval
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graphrag_vector_storage (namespace, vector_id, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (namespace, vector_id)
                DO UPDATE SET embedding = $3, metadata = $4, updated_at = NOW()
                """,
                self.namespace, vector_id, embedding_array, json.dumps(metadata)
            )
    
    async def query(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query for similar vectors using cosine similarity"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_func([query])
            if not query_embedding or len(query_embedding) == 0:
                logger.error(f"Failed to generate query embedding for: {query}")
                return []
            query_vec = query_embedding[0]
            
            async with self.db_pool.acquire() as conn:
                # Calculate cosine similarity in PostgreSQL
                # Note: This is not as efficient as pgvector but works without extensions
                rows = await conn.fetch(
                    """
                    WITH similarities AS (
                        SELECT 
                            vector_id,
                            metadata,
                            embedding,
                            -- Cosine similarity calculation
                            (
                                SELECT SUM(a.val * b.val) / (
                                    SQRT(SUM(a.val * a.val)) * SQRT(SUM(b.val * b.val))
                                )
                                FROM unnest(embedding) WITH ORDINALITY AS a(val, idx)
                                JOIN unnest($2::float8[]) WITH ORDINALITY AS b(val, idx) 
                                    ON a.idx = b.idx
                            ) AS similarity
                        FROM graphrag_vector_storage
                        WHERE namespace = $1
                    )
                    SELECT vector_id, metadata, similarity
                    FROM similarities
                    WHERE similarity IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT $3
                    """,
                    self.namespace, query_vec, top_k
                )
                
                results = []
                for row in rows:
                    result = dict(row['metadata'])
                    result['id'] = row['vector_id']
                    result['similarity'] = float(row['similarity'])
                    results.append(result)
                
                logger.info(f"Vector query returned {len(results)} results for namespace {self.namespace}")
                return results
                
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return []
    
    async def drop(self) -> None:
        """Clear all vectors for this namespace"""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM graphrag_vector_storage WHERE namespace = $1",
                self.namespace
            )
            deleted_count = int(result.split()[-1])
            logger.info(f"Dropped {deleted_count} vectors for namespace {self.namespace}")

    async def index_start_callback(self):
        """Called when indexing starts - no-op for PostgreSQL"""
        pass


class PostgreSQLGraphStorage(BaseGraphStorage):
    """PostgreSQL-based graph storage backend"""
    
    def __init__(self, namespace: str, global_config: dict):
        super().__init__()
        self.namespace = namespace
        self.global_config = global_config
        
        # Import db_manager from the database module
        try:
            from database import db_manager
            self.db_pool = db_manager.pool
        except ImportError:
            raise RuntimeError("Database manager not available. Cannot initialize PostgreSQL storage.")
        
        logger.info(f"PostgreSQL Graph Storage initialized for namespace: {namespace}")
    
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM graphrag_graph_nodes WHERE namespace = $1 AND node_id = $2",
                self.namespace, node_id
            )
            return row is not None
    
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM graphrag_graph_edges WHERE namespace = $1 AND source_id = $2 AND target_id = $3",
                self.namespace, source_node_id, target_node_id
            )
            return row is not None
    
    async def get_node(self, node_id: str) -> Union[Dict[str, Any], None]:
        """Get node data"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT properties FROM graphrag_graph_nodes WHERE namespace = $1 AND node_id = $2",
                self.namespace, node_id
            )
            if row:
                return dict(row['properties'])
            return None
    
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of edges) for a node"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) as degree
                FROM graphrag_graph_edges 
                WHERE namespace = $1 AND (source_id = $2 OR target_id = $2)
                """,
                self.namespace, node_id
            )
            return row['degree'] if row else 0
    
    async def edge_degree(self, source_node_id: str, target_node_id: str) -> int:
        """Get edge properties (returns 1 if exists, 0 if not)"""
        exists = await self.has_edge(source_node_id, target_node_id)
        return 1 if exists else 0
    
    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[Dict[str, Any], None]:
        """Get edge data"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT properties FROM graphrag_graph_edges WHERE namespace = $1 AND source_id = $2 AND target_id = $3",
                self.namespace, source_node_id, target_node_id
            )
            if row:
                return dict(row['properties'])
            return None
    
    async def get_node_edges(self, node_id: str) -> List[Tuple[str, str]]:
        """Get all edges connected to a node"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_id, target_id 
                FROM graphrag_graph_edges 
                WHERE namespace = $1 AND (source_id = $2 OR target_id = $2)
                """,
                self.namespace, node_id
            )
            return [(row['source_id'], row['target_id']) for row in rows]
    
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Insert or update a node"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graphrag_graph_nodes (namespace, node_id, properties)
                VALUES ($1, $2, $3)
                ON CONFLICT (namespace, node_id)
                DO UPDATE SET properties = $3, updated_at = NOW()
                """,
                self.namespace, node_id, json.dumps(node_data)
            )
    
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]) -> None:
        """Insert or update an edge"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graphrag_graph_edges (namespace, source_id, target_id, properties)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (namespace, source_id, target_id)
                DO UPDATE SET properties = $4, updated_at = NOW()
                """,
                self.namespace, source_node_id, target_node_id, json.dumps(edge_data)
            )
    
    async def embed_nodes(self, algorithm: str) -> List[List[float]]:
        """Embed graph nodes using specified algorithm - not implemented"""
        logger.warning("Graph node embedding not implemented for PostgreSQL backend")
        return []
    
    async def drop(self) -> None:
        """Clear all graph data for this namespace"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Delete edges first due to foreign key constraints
                edges_result = await conn.execute(
                    "DELETE FROM graphrag_graph_edges WHERE namespace = $1",
                    self.namespace
                )
                # Delete nodes
                nodes_result = await conn.execute(
                    "DELETE FROM graphrag_graph_nodes WHERE namespace = $1",
                    self.namespace
                )
                
                edges_deleted = int(edges_result.split()[-1])
                nodes_deleted = int(nodes_result.split()[-1])
                logger.info(f"Dropped {nodes_deleted} nodes and {edges_deleted} edges for namespace {self.namespace}")

    async def index_start_callback(self):
        """Called when indexing starts - no-op for PostgreSQL"""
        pass