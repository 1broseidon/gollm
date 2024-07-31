package ollama

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
	"strings"

	"github.com/1broseidon/gollm/models"
)

// OllamaProvider implements the Ollama-specific functionality
type OllamaProvider struct {
	baseURL string
	client  *http.Client
}

// NewOllamaProvider creates a new Ollama provider
func NewOllamaProvider() (*OllamaProvider, error) {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		return nil, errors.New("OLLAMA_BASE_URL environment variable is not set")
	}

	return &OllamaProvider{
		baseURL: baseURL,
		client:  &http.Client{},
	}, nil
}

// GenerateCompletion generates a completion using the specified Ollama model
func (p *OllamaProvider) GenerateCompletion(ctx context.Context, modelName string, input models.CompletionInput) (*models.CompletionResponse, error) {
	url := fmt.Sprintf("%s/api/generate", strings.TrimSuffix(p.baseURL, "/"))

	requestBody := map[string]interface{}{
		"model":  modelName,
		"prompt": input.Messages[len(input.Messages)-1].Content,
		"stream": false,
	}

	if input.MaxTokens > 0 {
		requestBody["options"] = map[string]interface{}{
			"num_predict": input.MaxTokens,
		}
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

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	response, ok := result["response"].(string)
	if !ok {
		return nil, errors.New("invalid response format")
	}

	promptEvalCount, _ := result["prompt_eval_count"].(float64)
	evalCount, _ := result["eval_count"].(float64)

	return &models.CompletionResponse{
		Text: response,
		Usage: &models.Usage{
			PromptTokens:     int(promptEvalCount),
			CompletionTokens: int(evalCount),
			TotalTokens:      int(promptEvalCount + evalCount),
		},
	}, nil
}

// GenerateCompletionStream generates a streaming completion using the specified Ollama model
func (p *OllamaProvider) GenerateCompletionStream(ctx context.Context, modelName string, input models.CompletionInput) (<-chan models.StreamingCompletionResponse, error) {
	url := fmt.Sprintf("%s/api/generate", strings.TrimSuffix(p.baseURL, "/"))

	requestBody := map[string]interface{}{
		"model":  modelName,
		"prompt": input.Messages[len(input.Messages)-1].Content,
		"stream": true,
	}

	if input.MaxTokens > 0 {
		requestBody["options"] = map[string]interface{}{
			"num_predict": input.MaxTokens,
		}
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

			var result map[string]interface{}
			if err := json.Unmarshal(line, &result); err != nil {
				streamChan <- models.StreamingCompletionResponse{Error: err}
				continue
			}

			response, ok := result["response"].(string)
			if ok {
				streamResponse := models.StreamingCompletionResponse{Text: response}

				promptEvalCount, _ := result["prompt_eval_count"].(float64)
				evalCount, _ := result["eval_count"].(float64)

				accumulatedUsage.PromptTokens = int(promptEvalCount)
				accumulatedUsage.CompletionTokens = int(evalCount)
				accumulatedUsage.TotalTokens = accumulatedUsage.PromptTokens + accumulatedUsage.CompletionTokens

				streamResponse.Usage = &accumulatedUsage

				done, ok := result["done"].(bool)
				if ok {
					streamResponse.Done = done
				}

				streamChan <- streamResponse

				if streamResponse.Done {
					return
				}
			}
		}
	}()

	return streamChan, nil
}

// Close closes the Ollama provider (no-op in this case)
func (p *OllamaProvider) Close() error {
	return nil
}

// GenerateEmbedding generates an embedding using the Ollama model (not implemented)
func (p *OllamaProvider) GenerateEmbedding(ctx context.Context, input string) ([]float32, error) {
	return nil, errors.New("embedding generation not implemented for Ollama provider")
}

// StartChat starts a new chat session (not implemented)
func (p *OllamaProvider) StartChat(modelName string) interface{} {
	return nil
}

// SendChatMessage sends a message to an existing chat session (not implemented)
func (p *OllamaProvider) SendChatMessage(ctx context.Context, session interface{}, message string) (*models.CompletionResponse, error) {
	return nil, errors.New("chat functionality not implemented for Ollama provider")
}
