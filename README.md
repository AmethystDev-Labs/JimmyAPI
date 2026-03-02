# ChatJimmy OpenAI Gateway

一个基于 Go + Gin 的高并发网关，把 OpenAI Chat Completions 请求转换为 JimmyAPI 请求格式。

## Features
- OpenAI-compatible endpoint: `POST /v1/chat/completions`
- Health check: `GET /healthz`
- Simple model list: `GET /v1/models`
- Request mapping:
  - `model -> chatOptions.selectedModel`
  - `system messages -> chatOptions.systemPrompt`
  - non-system messages -> `messages`
- Supports both `stream=true` and `stream=false`
- Strips Jimmy stats suffix: `<|stats|>...<|/stats|>`
- Reused HTTP transport and connection pool for high load

## Requirements
- Go 1.22+

## Run
```bash
go mod tidy
go run .
```

## Build
```bash
go build ./...
```

## Environment Variables
- `PORT`: listen port, default `8080`
- `JIMMY_BASE_URL`: default `https://chatjimmy.ai`
- `JIMMY_CHAT_PATH`: default `/api/chat`
- `JIMMY_TOP_K`: default `8`
- `JIMMY_REQUEST_TIMEOUT`: upstream timeout in seconds, default `90`
- `JIMMY_AUTH_HEADER`: optional upstream auth header name
- `JIMMY_AUTH_VALUE`: optional upstream auth header value

## Request Example (Non-stream)
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"llama3.1-8B",
    "stream":false,
    "messages":[
      {"role":"system","content":"You are a concise assistant."},
      {"role":"user","content":"Hello"}
    ]
  }'
```

## Request Example (Stream)
```bash
curl -sN http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"llama3.1-8B",
    "stream":true,
    "messages":[
      {"role":"user","content":"Say hi"}
    ]
  }'
```

## License
MIT
