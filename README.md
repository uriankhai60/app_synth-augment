# app_synth-augment

## 설치
```bash
uv venv venv -p=3.10.11
source venv/bin/activate
uv pip install -r requirements.txt
```

## 준비
```bash
touch .env
```
`.env` 파일에 아래 항목을 채우세요  
```text
HF_TOKEN="당신의 허깅페이스토큰"
FAL_KEY="당신의 FAL 서비스 키"
OPENAI_API_KEY="당신의 GPT API키"
LANGCHAIN_API_KEY="당신의 랭스미스 키"
GEMINI_API_KEY="당신의 제미나이 키"
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```