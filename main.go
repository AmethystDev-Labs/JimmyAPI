package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

type OpenAIChatRequest struct {
	Model    string          `json:"model"`
	Messages []OpenAIMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type OpenAIMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type JimmyRequest struct {
	Messages    []JimmyMessage   `json:"messages"`
	ChatOptions JimmyChatOptions `json:"chatOptions"`
	Attachment  *json.RawMessage `json:"attachment"`
}

type JimmyMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type JimmyChatOptions struct {
	SelectedModel string `json:"selectedModel"`
	SystemPrompt  string `json:"systemPrompt"`
	TopK          int    `json:"topK"`
}

type JimmyStats struct {
	Done         bool   `json:"done"`
	DoneReason   string `json:"done_reason"`
	Status       int    `json:"status"`
	TotalTokens  int    `json:"total_tokens"`
	PrefillToken int    `json:"prefill_tokens"`
	DecodeTokens int    `json:"decode_tokens"`
}

type OpenAIChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int                  `json:"index"`
	Message      *OpenAIResponseMsg   `json:"message,omitempty"`
	Delta        *OpenAIResponseDelta `json:"delta,omitempty"`
	FinishReason string               `json:"finish_reason,omitempty"`
}

type OpenAIResponseMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIResponseDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type GatewayConfig struct {
	Port            string
	JimmyBaseURL    string
	JimmyChatPath   string
	JimmyTopK       int
	RequestTimeout  time.Duration
	JimmyAuthHeader string
	JimmyAuthValue  string
}

func main() {
	cfg := loadConfig()

	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(gin.Logger())

	client := &http.Client{
		Timeout: cfg.RequestTimeout,
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   5 * time.Second,
				KeepAlive: 60 * time.Second,
			}).DialContext,
			MaxIdleConns:          2048,
			MaxIdleConnsPerHost:   1024,
			IdleConnTimeout:       120 * time.Second,
			TLSHandshakeTimeout:   5 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	}

	r.GET("/healthz", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	r.GET("/v1/models", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"object": "list",
			"data": []gin.H{{
				"id":       "jimmy-proxy-model",
				"object":   "model",
				"owned_by": "chatjimmy",
			}},
		})
	})

	r.POST("/v1/chat/completions", func(c *gin.Context) {
		handleChatCompletions(c, cfg, client)
	})

	addr := ":" + cfg.Port
	log.Printf("chatjimmy openai-gateway listening on %s", addr)
	if err := r.Run(addr); err != nil {
		log.Fatal(err)
	}
}

func loadConfig() GatewayConfig {
	port := getEnv("PORT", "8080")
	base := strings.TrimRight(getEnv("JIMMY_BASE_URL", "https://chatjimmy.ai"), "/")
	path := getEnv("JIMMY_CHAT_PATH", "/api/chat")
	topK := getEnvInt("JIMMY_TOP_K", 8)
	timeoutSec := getEnvInt("JIMMY_REQUEST_TIMEOUT", 90)

	return GatewayConfig{
		Port:            port,
		JimmyBaseURL:    base,
		JimmyChatPath:   path,
		JimmyTopK:       topK,
		RequestTimeout:  time.Duration(timeoutSec) * time.Second,
		JimmyAuthHeader: os.Getenv("JIMMY_AUTH_HEADER"),
		JimmyAuthValue:  os.Getenv("JIMMY_AUTH_VALUE"),
	}
}

func handleChatCompletions(c *gin.Context, cfg GatewayConfig, client *http.Client) {
	var req OpenAIChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, openAIError("invalid_request_error", "invalid JSON body"))
		return
	}
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, openAIError("invalid_request_error", "model is required"))
		return
	}
	if len(req.Messages) == 0 {
		c.JSON(http.StatusBadRequest, openAIError("invalid_request_error", "messages is required"))
		return
	}

	jimmyReq, err := mapOpenAIToJimmy(req, cfg.JimmyTopK)
	if err != nil {
		c.JSON(http.StatusBadRequest, openAIError("invalid_request_error", err.Error()))
		return
	}

	upstreamText, stats, err := callJimmy(c.Request.Context(), client, cfg, jimmyReq, c.GetHeader("Authorization"))
	if err != nil {
		c.JSON(http.StatusBadGateway, openAIError("upstream_error", err.Error()))
		return
	}

	cleanContent := strings.TrimSpace(upstreamText)
	respID := "chatcmpl-" + randHex(12)
	created := time.Now().Unix()

	if req.Stream {
		streamOpenAI(c, respID, created, req.Model, cleanContent)
		return
	}

	finishReason := "stop"
	if stats.DoneReason != "" {
		finishReason = stats.DoneReason
	}

	resp := OpenAIChatResponse{
		ID:      respID,
		Object:  "chat.completion",
		Created: created,
		Model:   req.Model,
		Choices: []OpenAIChoice{{
			Index: 0,
			Message: &OpenAIResponseMsg{
				Role:    "assistant",
				Content: cleanContent,
			},
			FinishReason: finishReason,
		}},
	}

	if stats.TotalTokens > 0 || stats.PrefillToken > 0 || stats.DecodeTokens > 0 {
		resp.Usage = &OpenAIUsage{
			PromptTokens:     stats.PrefillToken,
			CompletionTokens: stats.DecodeTokens,
			TotalTokens:      max(stats.TotalTokens, stats.PrefillToken+stats.DecodeTokens),
		}
	}

	c.JSON(http.StatusOK, resp)
}

func mapOpenAIToJimmy(req OpenAIChatRequest, topK int) (JimmyRequest, error) {
	systemParts := make([]string, 0, 2)
	messages := make([]JimmyMessage, 0, len(req.Messages))

	for _, m := range req.Messages {
		content, err := normalizeContent(m.Content)
		if err != nil {
			return JimmyRequest{}, fmt.Errorf("invalid content for role %s: %w", m.Role, err)
		}
		if strings.TrimSpace(content) == "" {
			continue
		}
		if strings.EqualFold(m.Role, "system") {
			systemParts = append(systemParts, content)
			continue
		}
		messages = append(messages, JimmyMessage{Role: m.Role, Content: content})
	}

	if len(messages) == 0 {
		return JimmyRequest{}, errors.New("at least one non-system message is required")
	}

	return JimmyRequest{
		Messages: messages,
		ChatOptions: JimmyChatOptions{
			SelectedModel: req.Model,
			SystemPrompt:  strings.Join(systemParts, "\n\n"),
			TopK:          topK,
		},
		Attachment: nil,
	}, nil
}

func callJimmy(ctx context.Context, client *http.Client, cfg GatewayConfig, payload JimmyRequest, inboundAuth string) (string, JimmyStats, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return "", JimmyStats{}, err
	}

	url := cfg.JimmyBaseURL + cfg.JimmyChatPath
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", JimmyStats{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	if cfg.JimmyAuthHeader != "" && cfg.JimmyAuthValue != "" {
		req.Header.Set(cfg.JimmyAuthHeader, cfg.JimmyAuthValue)
	} else if inboundAuth != "" {
		req.Header.Set("Authorization", inboundAuth)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", JimmyStats{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		rawErr, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return "", JimmyStats{}, fmt.Errorf("jimmy upstream status %d: %s", resp.StatusCode, strings.TrimSpace(string(rawErr)))
	}

	text, err := readStreamLikeText(resp.Body)
	if err != nil {
		return "", JimmyStats{}, err
	}

	content, stats := splitStats(text)
	return content, stats, nil
}

func readStreamLikeText(r io.Reader) (string, error) {
	var b strings.Builder
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
		if line == "" || line == "[DONE]" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(line)
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}
	return b.String(), nil
}

func splitStats(raw string) (string, JimmyStats) {
	start := strings.Index(raw, "<|stats|>")
	end := strings.Index(raw, "<|/stats|>")
	if start < 0 || end < 0 || end <= start {
		return raw, JimmyStats{}
	}

	content := strings.TrimSpace(raw[:start])
	statsJSON := strings.TrimSpace(raw[start+len("<|stats|>") : end])
	var stats JimmyStats
	_ = json.Unmarshal([]byte(statsJSON), &stats)
	return content, stats
}

func streamOpenAI(c *gin.Context, id string, created int64, model string, content string) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	writer := c.Writer
	flush, ok := writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, openAIError("server_error", "streaming not supported"))
		return
	}

	writeChunk := func(obj OpenAIChatResponse) bool {
		payload, err := json.Marshal(obj)
		if err != nil {
			return false
		}
		if _, err = writer.Write([]byte("data: ")); err != nil {
			return false
		}
		if _, err = writer.Write(payload); err != nil {
			return false
		}
		if _, err = writer.Write([]byte("\n\n")); err != nil {
			return false
		}
		flush.Flush()
		return true
	}

	chunkRole := OpenAIChatResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []OpenAIChoice{{
			Index: 0,
			Delta: &OpenAIResponseDelta{Role: "assistant"},
		}},
	}
	if !writeChunk(chunkRole) {
		return
	}

	chunkContent := OpenAIChatResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []OpenAIChoice{{
			Index: 0,
			Delta: &OpenAIResponseDelta{Content: content},
		}},
	}
	if !writeChunk(chunkContent) {
		return
	}

	chunkEnd := OpenAIChatResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []OpenAIChoice{{
			Index:        0,
			Delta:        &OpenAIResponseDelta{},
			FinishReason: "stop",
		}},
	}
	if !writeChunk(chunkEnd) {
		return
	}

	_, _ = writer.Write([]byte("data: [DONE]\n\n"))
	flush.Flush()
}

func normalizeContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}

	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		return asString, nil
	}

	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err == nil {
		segments := make([]string, 0, len(parts))
		for _, p := range parts {
			if t, ok := p["type"].(string); ok && t == "text" {
				if txt, ok := p["text"].(string); ok && txt != "" {
					segments = append(segments, txt)
				}
			}
		}
		return strings.Join(segments, "\n"), nil
	}

	return "", errors.New("content must be string or text-part array")
}

func openAIError(errType, message string) gin.H {
	return gin.H{
		"error": gin.H{
			"message": message,
			"type":    errType,
		},
	}
}

func getEnv(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func randHex(n int) string {
	if n <= 0 {
		return ""
	}
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
