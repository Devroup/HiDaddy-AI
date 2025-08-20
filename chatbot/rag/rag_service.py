"""
RAG 서비스 핵심 모듈

전체 RAG 워크플로우 관리 및 컨텍스트 생성을 담당
"""

import os
import re
import logging
from typing import List, Dict, Any
from pinecone import Pinecone
from openai import OpenAI
from .source_manager import SourceManager

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 검색 및 컨텍스트 생성 서비스"""
    
    def __init__(self):
        # 환경변수 로드
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_host = os.getenv("PINECONE_HOST") 
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "hidaddy")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "pregnancy-guide")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # 출처 관리자 초기화
        self.source_manager = SourceManager()
        
        # API 클라이언트 초기화
        self._init_clients()
    
    def _init_clients(self):
        """API 클라이언트들을 초기화합니다."""
        # Pinecone 초기화
        if self.pinecone_api_key and self.pinecone_host:
            try:
                self.pc = Pinecone(api_key=self.pinecone_api_key)
                self.index = self.pc.Index(name=self.index_name, host=self.pinecone_host)
                logger.info("Pinecone 연결 성공")
            except Exception as e:
                logger.error(f"Pinecone 연결 실패: {e}")
                self.index = None
        else:
            logger.warning("Pinecone 설정이 없습니다.")
            self.index = None
            
        # OpenAI 초기화
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            logger.warning("OpenAI API 키가 설정되지 않았습니다.")
            self.openai_client = None
    
    async def search_documents(self, query: str, top_k: int = 3, namespace: str = None) -> List[Dict[str, Any]]:
        """
        사용자 질문에 관련된 문서를 검색합니다.
        
        Args:
            query: 검색할 질문
            top_k: 반환할 문서 개수
            namespace: 사용할 네임스페이스 (None이면 기본값 사용)
            
        Returns:
            검색된 문서 리스트
        """
        if not self.index or not self.openai_client:
            logger.warning("RAG 서비스가 초기화되지 않았습니다.")
            return []
            
        try:
            # 질문을 임베딩으로 변환
            embedding_response = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Pinecone에서 유사한 문서 검색
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace or self.namespace,
                include_metadata=True
            )
            
            # 유사도 임계값 적용하여 결과 필터링
            documents = []
            for match in search_results.matches:
                if match.score > 0.4:
                    documents.append({
                        "content": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", ""),
                        "score": match.score
                    })
            
            logger.info(f"검색 완료: {len(documents)}개 문서 발견")
            return documents
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def build_context_prompt(self, user_query: str, documents: List[Dict[str, Any]]) -> str:
        """
        검색된 문서를 바탕으로 강화된 프롬프트를 생성합니다.
        """
        if not documents:
            return self._get_base_prompt(user_query)
        
        # 문서 내용을 컨텍스트로 구성
        context_parts = [doc["content"] for doc in documents if doc["content"].strip()]
        context_text = "\n\n".join(context_parts)
        
        return f"""당신은 임신, 출산, 육아를 준비하는 아빠들을 도와주는 전문 챗봇입니다.
                - 당신의 이름은 "아이키퍼"입니다.
                - 대화는 항상 한국어로만 진행합니다.
                - 사용자는 임신, 출산, 육아를 앞두고 있거나 진행 중인 **아빠**입니다.
                - 답변할 때 항상 아빠에게 말하듯 답변해주세요.
                - 임산부에 대해서 말할 때는 반드시 **"아내분"** 혹은 **"아내"** 라는 표현을 사용해주세요.

                [참고 문서 정보]
                {context_text}

                위 참고 문서의 정보를 우선적으로 활용하여 답변해주세요. 
                문서에 정확한 정보가 있다면 그것을 바탕으로 답변하고, 
                문서에 없는 내용이라면 일반적인 임신·출산·육아 상식으로 도움을 주세요.

                - 대답할 때는 친근하고 다정하며 존댓말을 사용합니다.
                - 사용자에게 안심과 신뢰감을 주는 톤으로 대답해주세요.
                - 답변은 되도록 간결하지만 핵심적인 정보를 전달합니다.
                - 너무 길지 않게 2~4문장 이내로 답변하세요.
                - 모르는 내용이거나 의학적 진단이 필요한 질문은 반드시 "죄송하지만, 그 부분은 전문가의 상담이 필요해요." 라고 대답하세요.
                - 산모와 아기를 생각하는 따뜻한 말 한마디를 마지막에 덧붙여 주세요.

                [사용자 질문]
                {user_query}"""
    
    def format_sources(self, documents: List[Dict[str, Any]]) -> str:
        """검색된 문서들의 출처 정보를 포맷팅합니다."""
        if not documents:
            return ""
        
        source_lines = []
        unique_sources = set()
        
        for doc in documents:
            source_file = doc.get("source", "")
            if source_file and source_file not in unique_sources:
                unique_sources.add(source_file)
                source_lines.append(self._format_single_source(source_file))
        
        if source_lines:
            return "<br><br>📚 참고자료:<br>" + "<br>".join([f"• {line}" for line in source_lines])
        return ""
    
    def _normalize_filename(self, filename: str) -> str:
        """파일명을 정규화합니다."""
        import unicodedata
        # 한글 자모 분리 문제 해결
        normalized = unicodedata.normalize('NFC', filename)
        return normalized
    
    def _find_similar_source(self, source_file: str) -> Dict[str, Any]:
        """유사한 파일명으로 출처를 찾습니다."""
        import difflib
        
        # sources.json의 모든 키 가져오기
        all_sources = list(self.source_manager.sources_data.keys())
        
        # 가장 유사한 키 찾기
        matches = difflib.get_close_matches(source_file, all_sources, n=1, cutoff=0.6)
        if matches:
            logger.info(f"유사한 파일명 찾음: '{source_file}' -> '{matches[0]}'")
            return self.source_manager.get_source_info(matches[0])
        
        return None
    
    def _format_single_source(self, source_file: str) -> str:
        """단일 출처 정보를 포맷팅합니다."""
        
        # 파일명 정규화 시도
        normalized_filename = self._normalize_filename(source_file)
        source_info = self.source_manager.get_source_info(normalized_filename)
        
        # 정규화해도 안 되면 sources.json의 모든 키와 유사도 검사
        if not source_info:
            source_info = self._find_similar_source(source_file)
        
        if not source_info:
            return f"출처: {source_file}"
        
        title = source_info.get("official_title", "")
        publisher = source_info.get("publisher", "")
        date = source_info.get("date", "")
        url = source_info.get("url", "")
        
        # 연도 추출
        year = ""
        if date:
            year_match = re.search(r'(\d{4})', date)
            if year_match:
                year = f"({year_match.group(1)})"
        
        source_line = f"{publisher}{year}. {title}"
        if url:
            source_line += f" - {url}"
        
        return source_line
    
    def _get_base_prompt(self, user_query: str) -> str:
        """기본 프롬프트 (RAG 검색 결과가 없을 때)"""
        return f"""당신은 임신, 출산, 육아를 준비하는 아빠들을 도와주는 전문 챗봇입니다.
                - 당신의 이름은 "아이키퍼"입니다.
                - 대화는 항상 한국어로만 진행합니다.
                - 사용자는 임신, 출산, 육아를 앞두고 있거나 진행 중인 **아빠**입니다.
                - 답변할 때 항상 아빠에게 말하듯 답변해주세요.
                - 임산부에 대해서 말할 때는 반드시 **"아내분"** 혹은 **"아내"** 라는 표현을 사용해주세요.
                - 임신, 출산, 산모 케어, 신생아 케어, 육아 관련 정보를 정확하게 전달합니다.
                - 대답할 때는 친근하고 다정하며 존댓말을 사용합니다.
                - 사용자에게 안심과 신뢰감을 주는 톤으로 대답해주세요.
                - 답변은 되도록 간결하지만 핵심적인 정보를 전달합니다.
                - 너무 길지 않게 2~4문장 이내로 답변하세요.
                - 모르는 내용이거나 의학적 진단이 필요한 질문은 반드시 "죄송하지만, 그 부분은 전문가의 상담이 필요해요." 라고 대답하세요.
                - 산모와 아기를 생각하는 따뜻한 말 한마디를 마지막에 덧붙여 주세요.

                [사용자 질문]
                {user_query}"""