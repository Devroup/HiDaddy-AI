"""
임베딩 관리 모듈

OpenAI 임베딩 생성 및 Pinecone 벡터 DB 업로드를 담당합니다.
"""

import os
import logging
from typing import List, Dict, Any
import time

from pinecone import Pinecone
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """임베딩 생성 및 벡터 DB 업로드 클래스"""
    
    def __init__(self):
        """초기화 및 API 클라이언트 설정"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_host = os.getenv("PINECONE_HOST")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "hidaddy")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # OpenAI 클라이언트 초기화
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            self.openai_client = None
            
        # Pinecone 클라이언트 초기화
        if self.pinecone_api_key and self.pinecone_host:
            try:
                self.pc = Pinecone(api_key=self.pinecone_api_key)
                self.index = self.pc.Index(name=self.index_name, host=self.pinecone_host)
                logger.info("Pinecone 연결 성공")
            except Exception as e:
                logger.error(f"Pinecone 연결 실패: {e}")
                self.index = None
        else:
            logger.error("Pinecone 설정이 누락되었습니다.")
            self.index = None
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        여러 텍스트에 대해 배치로 임베딩을 생성합니다.
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            임베딩 벡터 리스트
        """
        if not self.openai_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return []
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    input=batch_texts,
                    model=self.embedding_model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # API 요청 제한 고려
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"배치 임베딩 생성 실패 (batch {i//batch_size + 1}): {e}")
                return []
        
        logger.info(f"배치 임베딩 생성 완료: {len(embeddings)}개")
        return embeddings
    
    def upsert_documents(self, documents: List[Dict[str, Any]], 
                        namespace: str = "default", batch_size: int = 100) -> bool:
        """
        문서들을 벡터 DB에 업로드합니다.
        
        Args:
            documents: 업로드할 문서 리스트 [{"id": str, "text": str, "metadata": dict}, ...]
            namespace: Pinecone 네임스페이스
            batch_size: 업로드 배치 크기
            
        Returns:
            성공 여부
        """
        if not self.index:
            logger.error("Pinecone 인덱스가 초기화되지 않았습니다.")
            return False
        
        if not documents:
            logger.warning("업로드할 문서가 없습니다.")
            return True
        
        try:
            # 1. 텍스트에서 임베딩 생성
            texts = [doc["text"] for doc in documents]
            embeddings = self.create_embeddings_batch(texts)
            
            if not embeddings:
                return False
            
            # 2. 벡터 데이터 준비
            vectors = []
            for doc, embedding in zip(documents, embeddings):
                vector_data = {
                    "id": doc["id"],
                    "values": embedding,
                    "metadata": {**doc["metadata"], "text": doc["text"]}
                }
                vectors.append(vector_data)
            
            # 3. 배치로 업로드
            success_count = 0
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                
                try:
                    self.index.upsert(vectors=batch_vectors, namespace=namespace)
                    success_count += len(batch_vectors)
                    logger.info(f"배치 업로드 완료: {len(batch_vectors)}개 벡터")
                    
                    # API 요청 제한 고려
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"배치 업로드 실패 (batch {i//batch_size + 1}): {e}")
                    continue
            
            logger.info(f"문서 업로드 완료: {success_count}/{len(documents)}개 성공")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"문서 업로드 실패: {e}")
            return False