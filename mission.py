from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import os
import logging
from dotenv import load_dotenv

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

@app.post("/generate-mission", response_model=MissionResponse)
async def generate_mission(req: MissionRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다.")

    # recent_diaries 리스트를 문자열로 합치기
    diary_text = "\n".join(req.recent_diaries) if req.recent_diaries else "최근 작성된 감정일기가 없습니다."
    print(diary_text);
    print(req.guide);

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
            - description: 미션 설명 (한 문장)
            - keywords: 행동 1개, 사물/장소 1개, 표정/제스처 1개 (리스트)
        - 세 키워드는 하나의 사진 안에서 자연스럽게 표현될 수 있도록 연관성 있게 구성합니다.
        - 표정/제스처는 사진에서 명확히 인식 가능한 요소로 작성합니다.
        - 각 키워드는 한 단어나 두 단어로 간결하게 작성합니다.
        - 사진 촬영으로 미션 수행을 검증할 수 있도록 시각적으로 명확히 표현 가능한 요소로만 구성합니다.
        - 임산부의 안전을 해치는 행위(과도한 운동, 의료적 시술 등)는 제외합니다.

        응답 예시(JSON):
        {{
            "title": "아내에게 꽃을 선물하세요.",
            "description": "오늘 하루 수고한 아내에게 꽃을 선물하며 따뜻한 시간을 보내세요.",
            "keywords": ["꽃", "아내", "미소"]
        }}
    """

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            if response.status_code != 200:
                logger.error(f"OpenAI API 오류: {response.status_code} {response.text}")
                raise HTTPException(status_code=500, detail="OpenAI API 호출 실패")

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # JSON 파싱 시도
            try:
                mission_data = json.loads(content)
                print(mission_data);
            except josn.JSONDecodeError:
                logger.error(f"GPT 응답이 JSON 형식이 아님: {content}")
                raise HTTPException(status_code=500, detail="GPT 응답 파싱 실패")
            
            return MissionResponse(
                title=mission_data.get("title", "제목 없음"),
                description=mission_data.get("description", "설명 없음"),
                keywords=mission_data.get("keywords", [])[:3]
            )

    except httpx.RequestError as e:
        logger.error(f"HTTP 요청 중 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")
