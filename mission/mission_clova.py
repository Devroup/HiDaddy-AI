from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import os
import logging
from dotenv import load_dotenv
from typing import Optional
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="미션 키워드 생성 API",
    description="감정일기를 기반으로 GPT를 통해 행동 미션 키워드 3개를 추천",
    version="1.0.0"
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# CLOVA 환경변수 (generate-mission 용)
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_MODEL = os.getenv("CLOVA_MODEL", "HCX-DASH-002")
CLOVA_BASE = os.getenv("CLOVA_BASE", "https://clovastudio.stream.ntruss.com")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시에는 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MissionRequest(BaseModel):
    recent_diaries: list[str] = []  # 최근 3일 일기 (없으면 빈 리스트)
    guide: str               # 주차별 가이드 텍스트

class MissionResponse(BaseModel):
    title: str
    description: str
    keywords: list[str]

@app.get("/health")
async def health_check():
    """상태 확인 엔드포인트"""
    return {
        "status": "healthy",
        "openai_key_configured": bool(OPENAI_API_KEY)
    }

@app.post("/mission/generate-mission", response_model=MissionResponse)
async def generate_mission(req: MissionRequest):
    if not CLOVA_API_KEY:
        raise HTTPException(status_code=500, detail="CLOVA API 키가 설정되지 않았습니다.")

    # recent_diaries 리스트를 문자열로 합치기
    diary_text = "\n".join(req.recent_diaries) if req.recent_diaries else "최근 작성된 감정일기가 없습니다."

    prompt = f"""
        당신은 임신, 출산, 육아를 준비하는 아빠들을 돕는 전문가입니다.

        최근 남편의 감정일기:
        ---
        {diary_text}
        ---

        임신 주차별 가이드:
        ---
        {req.guide}
        ---

        위 자료를 바탕으로, 임신 중인 아내를 위해 남편이 오늘 실천하면 좋은 '행동 미션'을 제안하세요.

        조건:
        - JSON 형식으로만 응답합니다. 추가 설명이나 서문은 포함하지 않습니다.
        - JSON 키: title, description, keywords
            - title: 아빠를 대상으로 한 미션 제목 (한 줄)
                - 예를 들어, "~~을 하기"의 형식으로 구성합니다.
                - 문장 마지막엔 "." 온점을 반드시 빼줍니다.
            - description: 미션 설명 (한 문장)
            - keywords: 행동 1개, 사물/장소 1개, 표정/제스처 1개 (리스트)

        키워드 작성 시 반드시 지켜야 할 규칙:
        - 세 키워드는 반드시 하나의 사진 안에서 '동시에' 포착 가능해야 합니다.
        - 행동 키워드는 사물/장소 키워드와 함께 찍을 수 있는 것이어야 합니다.
            예: "안아주기"와 "침실"은 가능, "산책하기"와 "꽃"은 동시 표현이 어려움
        - 사물/장소는 행동을 하는 동안 자연스럽게 프레임에 잡힐 수 있어야 합니다.
            예: "손잡기"+"소파"는 가능, "요리하기"+"아내"는 대상이 중복되어 부적절
        - 표정/제스처는 행동을 수행하는 사람(주로 남편)이 보일 수 있는 것이어야 합니다.
            예: "미소", "포옹", "하트 손가락" 등
        - 각 키워드는 반드시 한 단어나 두 단어로 간결하게 작성합니다.
        - 사진 한 장에서 세 키워드를 모두 확인할 수 없다면, 해당 미션은 부적절합니다.
        - 임산부의 안전을 해치는 행위(과도한 운동, 의료적 시술 등)는 제외합니다.

        응답 예시(JSON):
        {{
            "title": "아내에게 꽃을 선물하세요.',
            "description": "오늘 하루 수고한 아내에게 꽃을 선물하며 따뜻한 시간을 보내세요.",
            "keywords": ["꽃 전달", "아내", "미소"]
        }}

        위 예시에서 세 키워드가 한 사진에 담기는 장면: 남편이 아내에게 꽃을 건네주며 미소 짓는 모습
    """

    # CLOVA v3 chat-completions 호출 준비
    url = f"{CLOVA_BASE}/v3/chat-completions/{CLOVA_MODEL}"

    body = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "너는 임신·출산·육아를 돕는 친절한 조언자다. 한국어로 간결하게 답하라."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "topP": 0.9,
        "topK": 0,
        "maxTokens": 300,
        "temperature": 0.5,
        "repetitionPenalty": 1.05,
        "stop": []
    }

    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=body, headers=headers)
            if response.status_code != 200:
                logger.error(f"CLOVA API 오류: {response.status_code} {response.text}")
                raise HTTPException(status_code=500, detail="CLOVA API 호출 실패")

            result = response.json()

            # 응답에서 텍스트 콘텐츠 추출
            content = ""
            try:
                content = (result.get("message", {}) or {}).get("content", "") or ""
                if not content and "choices" in result:
                    for ch in result["choices"]:
                        msg = (ch.get("message") or {})
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                            content += msg["content"]
                if not content:
                    content = (
                        (result.get("result", {}).get("message", {}) or {}).get("content", "") or
                        result.get("result", {}).get("output", {}).get("text") or
                        result.get("output_text") or
                        ""
                    )
            except Exception:
                content = ""

            if not content:
                logger.error(f"CLOVA 응답에서 콘텐츠 추출 실패: {result}")
                raise HTTPException(status_code=500, detail="AI 응답 처리 실패")

            content = content.strip()

            # JSON 파싱 시도 (```json 코드펜스 제거 처리 포함)
            try:
                if content.startswith("```"):
                    content = content[content.find("{"):content.rfind("}")+1]
                mission_data = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"AI 응답이 JSON 형식이 아님: {content}")
                raise HTTPException(status_code=500, detail="AI 응답 파싱 실패")

            return MissionResponse(
                title=mission_data.get("title", "제목 없음"),
                description=mission_data.get("description", "설명 없음"),
                keywords=mission_data.get("keywords", [])[:3]
            )

    except httpx.RequestError as e:
        logger.error(f"HTTP 요청 중 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")

# === 요청 DTO ===
class MissionAnalysisAiRequest(BaseModel):
    title: str
    description: str
    keyword1: str
    keyword2: str
    keyword3: str
    imageUrl: Optional[str] = None

# === 응답 DTO ===
class MissionAnalysisAiResponse(BaseModel):
    keyword1: bool
    keyword2: bool
    keyword3: bool
    result: str      # PASS 또는 FAIL
    reason: str      # 판정 사유


@app.post("/mission/analyze-photo", response_model=MissionAnalysisAiResponse)
async def analyze_photo(request: MissionAnalysisAiRequest):
    """
    CLOVA를 이용한 미션 사진 판독 API
    """
    try:
        # 입력 검증: imageUrl 필수
        if not request.imageUrl or not isinstance(request.imageUrl, str) or not request.imageUrl.strip():
            raise HTTPException(status_code=422, detail="imageUrl은 유효한 URL 문자열이어야 합니다.")

        # 판독 프롬프트 생성
        prompt = f"""
        다음은 미션 사진 판독 요청입니다:
        - 미션 제목: {request.title}
        - 미션 설명: {request.description}
        - 키워드: {request.keyword1}, {request.keyword2}, {request.keyword3}

        아래 이미지를 보고 각 키워드가 사진과 부합하는지 판단하세요.

        출력은 반드시 아래 JSON 형식으로만 하세요:
        {{
            "keyword1": true/false,
            "keyword2": true/false,
            "keyword3": true/false,
            "result": "PASS" 또는 "FAIL",
            "reason": "한 문장으로 간단히 판정 사유 설명"
        }}

        PASS 기준: 키워드 중 2개 이상이 true일 경우 PASS, 아니면 FAIL.
        """

        # CLOVA v3 chat-completions 호출 준비 (이미지 입력 포함)
        url = f"{CLOVA_BASE}/v3/chat-completions/{CLOVA_MODEL}"
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "너는 임신·출산·육아 미션 사진을 판독하는 심사관이다. 요구된 JSON 형식만 출력하라."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "imageUrl": {"url": request.imageUrl}}
                    ]
                }
            ],
            "topP": 0.9,
            "topK": 0,
            "maxTokens": 300,
            "temperature": 0.0,
            "repetitionPenalty": 1.05,
            "stop": []
        }

        headers = {
            "Authorization": f"Bearer {CLOVA_API_KEY}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=body, headers=headers)

        if response.status_code != 200:
            logger.error(f"CLOVA API 오류: {response.status_code} {response.text}")
            raise HTTPException(status_code=500, detail="CLOVA API 호출 실패 ㅇ")

        result = response.json()

        # 응답 텍스트 추출
        content = ""
        try:
            content = (result.get("message", {}) or {}).get("content", "") or ""
            if not content and "choices" in result:
                for ch in result["choices"]:
                    msg = (ch.get("message") or {})
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        content += msg["content"]
            if not content:
                content = (
                    (result.get("result", {}).get("message", {}) or {}).get("content", "") or
                    result.get("result", {}).get("output", {}).get("text") or
                    result.get("output_text") or
                    ""
                )
        except Exception:
            content = ""

        if not content:
            logger.error(f"CLOVA 응답에서 콘텐츠 추출 실패: {result}")
            raise HTTPException(status_code=500, detail="AI 응답 처리 실패")

        logger.info(f"CLOVA 응답 원본: {content[:200]}")

        # JSON 파싱 (코드펜스 제거 대응)
        try:
            if content.startswith("```"):
                content = content[content.find("{"):content.rfind("}")+1]
            mission_data = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"AI 응답이 JSON 형식이 아님: {content}")
            raise HTTPException(status_code=500, detail="AI 응답 파싱 실패")

        # 응답 DTO 변환
        return MissionAnalysisAiResponse(
            keyword1=mission_data.get("keyword1", False),
            keyword2=mission_data.get("keyword2", False),
            keyword3=mission_data.get("keyword3", False),
            result=mission_data.get("result", "FAIL"),
            reason=mission_data.get("reason", "판정 사유 없음")
        )

    except Exception as e:
        logger.error(f"미션 사진 판독 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="AI 판독 중 오류 발생")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 6000))
    uvicorn.run(app, host="0.0.0.0", port=port)