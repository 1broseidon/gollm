package openai

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

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// OpenAIProvider implements the OpenAI-specific functionality
type OpenAIProvider struct {
	apiKey string
	client *http.Client
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider() (*OpenAIProvider, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

	return &OpenAIProvider{
		apiKey: apiKey,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
}

// GenerateCompletion generates a completion using the specified OpenAI model
func (p *OpenAIProvider) GenerateCompletion(ctx context.Context, modelName string, input models.CompletionInput) (*models.CompletionResponse, error) {
	url := "https://api.openai.com/v1/chat/completions"

	requestBody := struct {
		Model       string               `json:"model"`
		Messages    []models.ChatMessage `json:"messages"`
		MaxTokens   int                  `json:"max_tokens"`
		Temperature float32              `json:"temperature"`
	}{
		Model:       modelName,
		Messages:    input.Messages,
		MaxTokens:   input.MaxTokens,
		Temperature: input.Temperature,
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
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API request failed: %w", fmt.Errorf("status code: %d, body: %s", resp.StatusCode, string(bodyBytes)))
	}

	// Log the response body for debugging
	bodyBytes, _ := io.ReadAll(resp.Body)
	fmt.Printf("OpenAI API Response: %s\n", string(bodyBytes))

	// Create a new reader with the body bytes
	resp.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, errors.New("invalid response format")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid choice format")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid message format")
	}

	content, ok := message["content"].(string)
	if !ok {
		return nil, errors.New("invalid content format")
	}

	usage, ok := result["usage"].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid usage format")
	}

	promptTokens, ok := usage["prompt_tokens"].(float64)
	if !ok {
		return nil, errors.New("invalid prompt_tokens format")
	}

	completionTokens, ok := usage["completion_tokens"].(float64)
	if !ok {
		return nil, errors.New("invalid completion_tokens format")
	}

	totalTokens, ok := usage["total_tokens"].(float64)
	if !ok {
		return nil, errors.New("invalid total_tokens format")
	}

	response := &models.CompletionResponse{
		Text: content,
		Usage: &models.Usage{
			PromptTokens:     int(promptTokens),
			CompletionTokens: int(completionTokens),
			TotalTokens:      int(totalTokens),
		},
	}

	return response, nil
}

// GenerateCompletionStream generates a streaming completion using the specified OpenAI model
func (p *OpenAIProvider) GenerateCompletionStream(ctx context.Context, modelName string, input models.CompletionInput) (<-chan models.StreamingCompletionResponse, error) {
	url := "https://api.openai.com/v1/chat/completions"

	requestBody := map[string]interface{}{
		"model":       modelName,
		"messages":    input.Messages,
		"max_tokens":  input.MaxTokens,
		"temperature": input.Temperature,
		"stream":      true,
		"stream_options": map[string]bool{
			"include_usage": true,
		},
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
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

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

			if len(line) == 0 {
				continue
			}

			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}

			data := bytes.TrimPrefix(line, []byte("data: "))
			if bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
				streamChan <- models.StreamingCompletionResponse{Done: true, Usage: &accumulatedUsage}
				return
			}

			var result map[string]interface{}
			if err := json.Unmarshal(data, &result); err != nil {
				// Skip this error for "[DONE]" message
				if !bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
					fmt.Printf("Error unmarshaling JSON: %v\nData: %s\n", err, string(data))
					streamChan <- models.StreamingCompletionResponse{Error: fmt.Errorf("error unmarshaling JSON: %v", err)}
				}
				continue
			}

			choices, ok := result["choices"].([]interface{})
			if !ok {
				// This might be the final usage chunk
				usage, ok := result["usage"].(map[string]interface{})
				if ok {
					promptTokens, _ := usage["prompt_tokens"].(float64)
					completionTokens, _ := usage["completion_tokens"].(float64)
					totalTokens, _ := usage["total_tokens"].(float64)

					accumulatedUsage = models.Usage{
						PromptTokens:     int(promptTokens),
						CompletionTokens: int(completionTokens),
						TotalTokens:      int(totalTokens),
					}
					streamChan <- models.StreamingCompletionResponse{Done: true, Usage: &accumulatedUsage}
					return
				}
				continue
			}

			if len(choices) == 0 {
				// This might be the final usage chunk
				usage, ok := result["usage"].(map[string]interface{})
				if ok {
					promptTokens, _ := usage["prompt_tokens"].(float64)
					completionTokens, _ := usage["completion_tokens"].(float64)
					totalTokens, _ := usage["total_tokens"].(float64)

					accumulatedUsage = models.Usage{
						PromptTokens:     int(promptTokens),
						CompletionTokens: int(completionTokens),
						TotalTokens:      int(totalTokens),
					}
					streamChan <- models.StreamingCompletionResponse{Done: true, Usage: &accumulatedUsage}
					return
				}
				continue
			}

			choice, ok := choices[0].(map[string]interface{})
			if !ok {
				err := fmt.Errorf("invalid choice format")
				fmt.Println(err)
				streamChan <- models.StreamingCompletionResponse{Error: err}
				continue
			}

			delta, ok := choice["delta"].(map[string]interface{})
			if !ok {
				err := fmt.Errorf("invalid delta format")
				fmt.Println(err)
				streamChan <- models.StreamingCompletionResponse{Error: err}
				continue
			}

			content, ok := delta["content"].(string)
			if ok {
				response := models.StreamingCompletionResponse{Text: content}

				// Check if this is the last chunk
				finishReason, ok := choice["finish_reason"].(string)
				if ok && finishReason != "" {
					response.Done = true
					response.Usage = &accumulatedUsage
				}

				// Update usage metadata if available
				if usageMetadata, ok := result["usageMetadata"].(map[string]interface{}); ok {
					promptTokenCount, _ := usageMetadata["promptTokenCount"].(float64)
					candidatesTokenCount, _ := usageMetadata["candidatesTokenCount"].(float64)
					totalTokenCount, _ := usageMetadata["totalTokenCount"].(float64)

					accumulatedUsage = models.Usage{
						PromptTokens:     int(promptTokenCount),
						CompletionTokens: int(candidatesTokenCount),
						TotalTokens:      int(totalTokenCount),
					}
					response.Usage = &accumulatedUsage
				}

				streamChan <- response

				if response.Done {
					return
				}
			}
		}
	}()

	return streamChan, nil
}

// Close closes the OpenAI provider (no-op in this case)
func (p *OpenAIProvider) Close() error {
	return nil
}

// GenerateEmbedding generates an embedding using the OpenAI model (not implemented)
func (p *OpenAIProvider) GenerateEmbedding(ctx context.Context, input string) ([]float32, error) {
	return nil, errors.New("embedding generation not implemented for OpenAI provider")
}

// StartChat starts a new chat session (not implemented)
func (p *OpenAIProvider) StartChat(modelName string) interface{} {
	return nil
}

// SendChatMessage sends a message to an existing chat session (not implemented)
func (p *OpenAIProvider) SendChatMessage(ctx context.Context, session interface{}, message string) (*models.CompletionResponse, error) {
	return nil, errors.New("chat functionality not implemented for OpenAI provider")
}
