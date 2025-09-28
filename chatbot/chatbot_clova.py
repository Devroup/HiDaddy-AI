"""
아이키퍼 챗봇 API

임신, 출산, 육아를 준비하는 아빠들을 위한 RAG 기반 챗봇 API
"""
import uuid
import httpx, json, os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os
import logging
from dotenv import load_dotenv

from rag.rag_service import RAGService

# 환경변수 로드 및 로깅 설정
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="아이키퍼 챗봇 API",
    description="임신, 출산, 육아를 준비하는 아빠들을 위한 챗봇 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경변수
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_MODEL   = os.getenv("CLOVA_MODEL", "HCX-DASH-002")
CLOVA_BASE    = os.getenv("CLOVA_BASE", "https://clovastudio.stream.ntruss.com")

# RAG 서비스 초기화
rag_service = RAGService()
logger.info("RAG 서비스 초기화 완료")


class ConnectionManager:
    """웹소켓 연결 관리"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"웹소켓 연결됨. 총 연결 수: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"웹소켓 연결 해제. 총 연결 수: {len(self.active_connections)}")


manager = ConnectionManager()


@app.get("/")
async def root():
    """헬스체크 엔드포인트"""
    return {"message": "아이키퍼 챗봇 API가 실행 중입니다!"}


@app.get("/health")
async def health_check():
    """상태 확인 엔드포인트"""
    return {
        "status": "healthy",
        "clova_key_configured": bool(CLOVA_API_KEY)
    }


async def send_error(websocket: WebSocket, message: str):
    """에러 메시지 전송"""
    error_data = {'type': 'error', 'content': message}
    await websocket.send_text(json.dumps(error_data))


async def send_chunk(websocket: WebSocket, content: str):
    """청크 데이터 전송"""
    chunk_data = {'type': 'chunk', 'content': content}
    await websocket.send_text(json.dumps(chunk_data))


async def send_end(websocket: WebSocket):
    """완료 신호 전송"""
    end_data = {'type': 'end', 'content': ''}
    await websocket.send_text(json.dumps(end_data))


async def get_enhanced_prompt(user_message: str):
    """RAG 검색을 통해 강화된 프롬프트 생성"""
    try:
        logger.info(f"RAG 검색 시작: {user_message[:50]}...")
        search_results = await rag_service.search_documents(user_message, top_k=3)
        
        if search_results:
            logger.info(f"RAG 검색 결과: {len(search_results)}개 문서 발견")
            return rag_service.build_context_prompt(user_message, search_results), search_results
        else:
            logger.info("RAG 검색 결과 없음, 기본 프롬프트 사용")
            return rag_service._get_base_prompt(user_message), []
            
    except Exception as e:
        logger.error(f"RAG 검색 실패: {e}, 기본 프롬프트 사용")
        fallback_prompt = f"""당신은 임신, 출산, 육아를 준비하는 아빠들을 도와주는 전문 챗봇입니다.
- 당신의 이름은 "아이키퍼"입니다.
- 대화는 항상 한국어로만 진행합니다.
- 사용자는 임신, 출산, 육아를 앞두고 있거나 진행 중인 **아빠**입니다.
- 답변할 때 항상 아빠에게 말하듯 답변해주세요.
- 임산부에 대해서 말할 때는 반드시 **"아내분"** 혹은 **"아내"** 라는 표현을 사용해주세요.

[사용자 질문]
{user_message}"""
        return fallback_prompt, []

async def chat_with_clova_stream(prompt: str, websocket: WebSocket, image_url: str | None = None):
    """
    CLOVA Studio v3 SSE 엔드포인트로 스트리밍 호출.
    주어진 prompt(텍스트)와 선택적 image_url을 이용한다.
    """
    url = f"{CLOVA_BASE}/v3/chat-completions/{CLOVA_MODEL}"

    # 메시지 포맷(v3): content는 배열이며 type 지정(text / image_url)
    user_content = []
    if image_url:
        user_content.append({
            "type": "image_url",
            "imageUrl": {"url": image_url}
        })
    user_content.append({"type": "text", "text": prompt})

    body = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "너는 임신·출산·육아를 돕는 친절한 조언자 '아이키퍼'다. 한국어로 간결하게 답하라."}]
            },
            {"role": "user", "content": user_content}
        ],
        "topP": 0.9,
        "topK": 0,
        "maxTokens": 800,
        "temperature": 0.5,
        "repetitionPenalty": 1.05,
        "stop": [],
        # 중요: 스트리밍은 Accept 헤더로 동작. body에 stream 플래그가 필요없다면 생략해도 됨.
    }

    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    err = (await resp.aread()).decode("utf-8", "ignore")
                    await send_error(websocket, f"CLOVA 오류 {resp.status_code}: {err[:200]}")
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload in ("", "[DONE]"):
                        logger.debug("CLOVA frame: [DONE]")
                        break

                    logger.debug("CLOVA frame raw: %s", payload[:500])

                    text_piece = ""
                    try:
                        obj = json.loads(payload)

                        # 1) 최상위 message.content
                        text_piece = (obj.get("message", {}) or {}).get("content", "") or ""

                        # 2) 다른 스키마들(보조)
                        if not text_piece and "choices" in obj:
                            for ch in obj["choices"]:
                                delta = ch.get("delta") or {}
                                msg = delta.get("message") or {}
                                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                                    text_piece += msg["content"]
                                content = delta.get("content")
                                if isinstance(content, list):
                                    for c in content:
                                        if isinstance(c, dict) and c.get("type") == "text":
                                            text_piece += c.get("text", "")
                        if not text_piece:
                            text_piece = (
                                (obj.get("result", {}).get("message", {}) or {}).get("content", "") or
                                obj.get("result", {}).get("output", {}).get("text") or
                                obj.get("output_text") or
                                ""
                            )

                    except json.JSONDecodeError:
                        text_piece = payload  # 텍스트 프레임이면 그대로

                    if text_piece:  # 빈 문자열 프레임은 건너뜀
                        logger.debug("CLOVA parsed text: %s", text_piece[:200])
                        await send_chunk(websocket, text_piece)

    except httpx.TimeoutException:
        await send_error(websocket, "응답 시간이 초과되었습니다. 다시 시도해주세요.")
    except Exception as e:
        logger.exception("CLOVA 호출 실패")
        await send_error(websocket, f"서버 오류: {e}")

async def process_chat_message(user_message: str, websocket: WebSocket):
    """채팅 메시지 처리"""
    if not CLOVA_API_KEY:
        await send_error(websocket, "CLOVA API 키가 설정되지 않았습니다.")
        return

    # RAG 강화 프롬프트 생성
    enhanced_prompt, search_results = await get_enhanced_prompt(user_message)

    # (필요 시) 이미지 URL을 파싱해서 넘겨도 됨: image_url = extract_from_user_message(...)
    await chat_with_clova_stream(enhanced_prompt, websocket)

    if search_results:
        source_info = rag_service.format_sources(search_results)
        if source_info:
            await send_chunk(websocket, source_info)

    await send_end(websocket)


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 메시지 수신
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get('message', '').strip()
            if not user_message:
                await send_error(websocket, "메시지가 필요합니다.")
                continue
            
            logger.info(f"채팅 요청: {user_message[:50]}...")
            
            # 메시지 처리
            await process_chat_message(user_message, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"웹소켓 오류: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)