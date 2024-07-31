package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/1broseidon/gollm/models"
)

// AnthropicProvider implements the Anthropic-specific functionality
type AnthropicProvider struct {
	apiKey string
	client *http.Client
}

// NewAnthropicProvider creates a new Anthropic provider
func NewAnthropicProvider() (*AnthropicProvider, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, errors.New("ANTHROPIC_API_KEY environment variable is not set")
	}

	return &AnthropicProvider{
		apiKey: apiKey,
		client: &http.Client{},
	}, nil
}

// GenerateCompletion generates a completion using the specified Anthropic model
func (p *AnthropicProvider) GenerateCompletion(ctx context.Context, modelName string, input models.CompletionInput) (*models.CompletionResponse, error) {
	url := "https://api.anthropic.com/v1/messages"

	requestBody := struct {
		Model     string               `json:"model"`
		Messages  []models.ChatMessage `json:"messages"`
		MaxTokens int                  `json:"max_tokens"`
	}{
		Model:     modelName,
		Messages:  input.Messages,
		MaxTokens: input.MaxTokens,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.Content) == 0 {
		return nil, errors.New("no content in response")
	}

	response := &models.CompletionResponse{
		Text: result.Content[0].Text,
		Usage: &models.Usage{
			PromptTokens:     result.Usage.InputTokens,
			CompletionTokens: result.Usage.OutputTokens,
			TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
		},
	}

	return response, nil
}

// GenerateCompletionStream generates a streaming completion using the specified Anthropic model
func (p *AnthropicProvider) GenerateCompletionStream(ctx context.Context, modelName string, input models.CompletionInput) (<-chan models.StreamingCompletionResponse, error) {
	url := "https://api.anthropic.com/v1/messages"

	requestBody := map[string]interface{}{
		"model":      modelName,
		"messages":   input.Messages,
		"max_tokens": input.MaxTokens,
		"stream":     true,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("API request failed with status code: %d", resp.StatusCode)
	}

	streamChan := make(chan models.StreamingCompletionResponse)

	go func() {
		defer resp.Body.Close()
		defer close(streamChan)

		reader := bufio.NewReader(resp.Body)
		var accumulatedText string
		var accumulatedUsage models.Usage

		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					return
				}
				streamChan <- models.StreamingCompletionResponse{Error: err}
				return
			}

			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}

			data := bytes.TrimPrefix(line, []byte("data: "))
			var event map[string]interface{}
			if err := json.Unmarshal(data, &event); err != nil {
				streamChan <- models.StreamingCompletionResponse{Error: err}
				continue
			}

			eventType, ok := event["type"].(string)
			if !ok {
				continue
			}

			switch eventType {
			case "message_start":
				message, ok := event["message"].(map[string]interface{})
				if !ok {
					continue
				}
				usage, ok := message["usage"].(map[string]interface{})
				if !ok {
					continue
				}
				inputTokens, ok := usage["input_tokens"].(float64)
				if !ok {
					continue
				}
				accumulatedUsage.PromptTokens = int(inputTokens)

			case "content_block_delta":
				delta, ok := event["delta"].(map[string]interface{})
				if !ok {
					continue
				}
				text, ok := delta["text"].(string)
				if !ok {
					continue
				}
				accumulatedText += text
				streamChan <- models.StreamingCompletionResponse{Text: text}

			case "message_delta":
				usage, ok := event["usage"].(map[string]interface{})
				if !ok {
					continue
				}
				outputTokens, ok := usage["output_tokens"].(float64)
				if !ok {
					continue
				}
				accumulatedUsage.CompletionTokens = int(outputTokens)
				accumulatedUsage.TotalTokens = accumulatedUsage.PromptTokens + accumulatedUsage.CompletionTokens

			case "message_stop":
				streamChan <- models.StreamingCompletionResponse{
					Text:  accumulatedText,
					Done:  true,
					Usage: &accumulatedUsage,
				}
				return
			}
		}
	}()

	return streamChan, nil
}

// Close closes the Anthropic provider (no-op in this case)
func (p *AnthropicProvider) Close() error {
	return nil
}

// GenerateEmbedding generates an embedding using the Anthropic model (not implemented)
func (p *AnthropicProvider) GenerateEmbedding(ctx context.Context, input string) ([]float32, error) {
	return nil, errors.New("embedding generation not implemented for Anthropic provider")
}

// StartChat starts a new chat session (not implemented)
func (p *AnthropicProvider) StartChat(modelName string) interface{} {
	return nil
}

// SendChatMessage sends a message to an existing chat session (not implemented)
func (p *AnthropicProvider) SendChatMessage(ctx context.Context, session interface{}, message string) (*models.CompletionResponse, error) {
	return nil, errors.New("chat functionality not implemented for Anthropic provider")
}
