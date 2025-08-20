"""
출처 관리 모듈

sources.json 파일을 통해 문서의 상세 출처 정보를 관리합니다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SourceManager:
    """문서 출처 정보 관리 클래스"""
    
    def __init__(self, sources_file: str = "data/sources.json"):
        """
        Args:
            sources_file: sources.json 파일 경로
        """
        self.sources_file = Path(sources_file)
        self.sources_data = self._load_sources()
    
    def _load_sources(self) -> Dict[str, Any]:
        """sources.json 파일을 로드합니다."""
        if not self.sources_file.exists():
            logger.warning(f"출처 파일이 없습니다: {self.sources_file}")
            return {}
        
        try:
            with open(self.sources_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"출처 파일 로드 실패: {e}")
            return {}
    
    def get_source_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        파일명으로 출처 정보를 조회합니다.
        
        Args:
            filename: 문서 파일명
            
        Returns:
            출처 정보 딕셔너리 또는 None
        """
        return self.sources_data.get(filename)