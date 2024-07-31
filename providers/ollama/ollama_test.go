package ollama

import (
	"context"
	"os"
	"testing"

	"github.com/1broseidon/gollm/models"
)

func TestOllamaProvider(t *testing.T) {
	// Skip the test if OLLAMA_BASE_URL is not set
	if os.Getenv("OLLAMA_BASE_URL") == "" {
		t.Skip("OLLAMA_BASE_URL not set, skipping Ollama provider test")
	}

	ctx := context.Background()

	provider, err := NewOllamaProvider()
	if err != nil {
		t.Fatalf("Failed to create Ollama provider: %v", err)
	}

	t.Run("GenerateCompletion", func(t *testing.T) {
		input := models.CompletionInput{
			Model: "llama3.1:latest",
			Messages: []models.ChatMessage{
				{Role: "user", Content: "Explain the concept of quantum computing in simple terms."},
			},
			MaxTokens:   100,
			Temperature: 0.7,
		}

		response, err := provider.GenerateCompletion(ctx, "llama3.1:latest", input)
		if err != nil {
			t.Fatalf("GenerateCompletion failed: %v", err)
		}

		if response.Text == "" {
			t.Error("Generated text is empty")
		}

		if response.Usage == nil {
			t.Error("Usage information is missing")
		} else {
			if response.Usage.PromptTokens == 0 {
				t.Error("Prompt token count is zero")
			}
			if response.Usage.CompletionTokens == 0 {
				t.Error("Completion token count is zero")
			}
			if response.Usage.TotalTokens == 0 {
				t.Error("Total token count is zero")
			}
		}
	})

	t.Run("GenerateCompletionStream", func(t *testing.T) {
		input := models.CompletionInput{
			Model: "llama3.1:latest",
			Messages: []models.ChatMessage{
				{Role: "user", Content: "Explain the concept of artificial intelligence in simple terms."},
			},
			MaxTokens:   100,
			Temperature: 0.7,
			Stream:      true,
		}

		streamChan, err := provider.GenerateCompletionStream(ctx, "llama3.1:latest", input)
		if err != nil {
			t.Fatalf("GenerateCompletionStream failed: %v", err)
		}

		var fullText string
		var lastChunk models.StreamingCompletionResponse
		for chunk := range streamChan {
			if chunk.Error != nil {
				t.Fatalf("Error in streaming: %v", chunk.Error)
			}
			if chunk.Text != "" {
				fullText += chunk.Text
			}
			lastChunk = chunk
			if chunk.Done {
				break
			}
		}

		if fullText == "" {
			t.Error("Generated streaming text is empty")
		}
		if lastChunk.Usage == nil {
			t.Error("Usage information was not received in the final stream chunk")
		} else {
			if lastChunk.Usage.PromptTokens == 0 {
				t.Error("Prompt token count is zero")
			}
			if lastChunk.Usage.CompletionTokens == 0 {
				t.Error("Completion token count is zero")
			}
			if lastChunk.Usage.TotalTokens == 0 {
				t.Error("Total token count is zero")
			}
		}
	})
}
