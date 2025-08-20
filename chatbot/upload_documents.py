#!/usr/bin/env python3
"""
PDF 문서 업로드 스크립트

지정된 폴더의 PDF 문서를 Pinecone에 업로드합니다.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

from rag.document_loader import DocumentLoader
from rag.embeddings import EmbeddingManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """필수 환경변수 확인"""
    load_dotenv()
    
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_HOST"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"다음 환경변수가 설정되지 않았습니다: {missing_vars}")
        sys.exit(1)


def validate_directory(data_dir: str):
    """데이터 디렉토리 유효성 검사"""
    dir_path = Path(data_dir)
    if not dir_path.exists():
        logger.error(f"디렉토리가 존재하지 않습니다: {data_dir}")
        sys.exit(1)
    
    pdf_files = list(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"PDF 파일이 없습니다: {data_dir}")
        return False
    
    logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")
    return True


def print_document_stats(documents):
    """문서별 청크 통계 출력"""
    doc_stats = {}
    for doc in documents:
        source = doc['metadata']['source']
        doc_stats[source] = doc_stats.get(source, 0) + 1
    
    logger.info("파일별 청크 수:")
    for source, count in doc_stats.items():
        logger.info(f"  - {source}: {count}개 청크")


def main():
    parser = argparse.ArgumentParser(description="PDF 문서를 Pinecone에 업로드")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/new",
        help="PDF 문서가 있는 디렉토리 경로"
    )
    parser.add_argument(
        "--namespace", 
        type=str, 
        default="pregnancy-guide",
        help="Pinecone 네임스페이스"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1200,
        help="텍스트 청킹 크기"
    )
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=150,
        help="청크 간 중복 크기"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=50,
        help="업로드 배치 크기"
    )
    
    args = parser.parse_args()
    
    # 환경 검증
    check_environment()
    if not validate_directory(args.data_dir):
        sys.exit(1)
    
    logger.info("=== PDF 문서 업로드 시작 ===")
    logger.info(f"디렉토리: {args.data_dir}")
    logger.info(f"네임스페이스: {args.namespace}")
    logger.info(f"청킹 설정: {args.chunk_size}자 (중복: {args.overlap}자)")
    
    try:
        # 초기화
        document_loader = DocumentLoader(args.chunk_size, args.overlap)
        embedding_manager = EmbeddingManager()
        
        # 문서 로드
        logger.info("PDF 문서 로드 및 전처리 중...")
        documents = document_loader.load_documents_from_directory(args.data_dir)
        
        if not documents:
            logger.error("처리된 문서가 없습니다.")
            sys.exit(1)
        
        logger.info(f"총 {len(documents)}개 청크 생성 완료")
        print_document_stats(documents)
        
        # Pinecone 업로드
        logger.info("Pinecone 업로드 중...")
        success = embedding_manager.upsert_documents(
            documents=documents,
            namespace=args.namespace,
            batch_size=args.batch_size
        )
        
        if success:
            logger.info("업로드 완료")
        else:
            logger.error("업로드 실패")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()