"""
문서 로더 모듈

PDF 문서를 로드하고 전처리하는 기능을 제공합니다.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from hashlib import sha1

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)


class DocumentLoader:
    """PDF 문서 로딩 및 전처리 클래스"""
    
    def __init__(self, chunk_size: int = 1200, overlap: int = 150):
        """
        Args:
            chunk_size: 텍스트 청킹 크기
            overlap: 청크 간 중복 크기
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def load_pdf(self, file_path: str) -> str:
        """
        PDF 파일에서 텍스트를 추출합니다.
        
        Args:
            file_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        if not PdfReader:
            logger.error("pypdf 패키지가 설치되지 않았습니다.")
            return ""
        
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return ""
        
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF 읽기 실패 {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 적절한 크기로 분할합니다.
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 텍스트 청크 리스트
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 문장 단위로 끊어지도록 조정
            if end < len(text):
                last_sentence_end = text.rfind('.', start, end)
                if last_sentence_end > start:
                    end = last_sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 다음 시작점 설정 (중복 고려)
            start = max(start + self.chunk_size - self.overlap, end)
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        PDF 문서를 로드하고 청킹하여 처리합니다.
        
        Args:
            file_path: PDF 파일 경로
            
        Returns:
            처리된 문서 청크 리스트
        """
        text = self.load_pdf(file_path)
        if not text:
            return []
        
        chunks = self.chunk_text(text)
        if not chunks:
            return []
        
        file_path = Path(file_path)
        
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_id = sha1(f"{file_path.name}_{idx}".encode()).hexdigest()
            
            chunk_data = {
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": file_path.name,
                    "title": file_path.stem,
                    "chunk_index": idx
                }
            }
            processed_chunks.append(chunk_data)
        
        logger.info(f"문서 처리 완료: {file_path.name}, {len(processed_chunks)}개 청크")
        return processed_chunks
    
    def load_documents_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        디렉토리에서 모든 PDF 문서를 로드하고 처리합니다.
        
        Args:
            directory_path: 문서 디렉토리 경로
            
        Returns:
            처리된 모든 문서 청크 리스트
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"디렉토리가 존재하지 않습니다: {directory_path}")
            return []
        
        all_chunks = []
        
        for file_path in directory.glob("*.pdf"):
            if file_path.is_file():
                try:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"문서 처리 실패 {file_path}: {e}")
                    continue
        
        logger.info(f"디렉토리 처리 완료: {len(all_chunks)}개 총 청크")
        return all_chunks