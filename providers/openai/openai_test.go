package openai

import (
	"context"
	"os"
	"testing"

	"github.com/1broseidon/gollm/models"
)

func TestOpenAIProvider(t *testing.T) {
	// Skip the test if OPENAI_API_KEY is not set
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set, skipping OpenAI provider test")
	}

	ctx := context.Background()

	provider, err := NewOpenAIProvider()
	if err != nil {
		t.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	t.Run("GenerateCompletion", func(t *testing.T) {
		input := models.CompletionInput{
			Model: "gpt-3.5-turbo",
			Messages: []models.ChatMessage{
				{Role: "user", Content: "Explain the concept of machine learning in simple terms."},
			},
			MaxTokens:   100,
			Temperature: 0.7,
		}

		response, err := provider.GenerateCompletion(ctx, "gpt-3.5-turbo", input)
		if err != nil {
			t.Fatalf("GenerateCompletion failed: %v\nInput: %+v", err, input)
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
			Model: "gpt-3.5-turbo",
			Messages: []models.ChatMessage{
				{Role: "user", Content: "Explain the concept of artificial intelligence in simple terms."},
			},
			MaxTokens:   100,
			Temperature: 0.7,
			Stream:      true,
		}


		streamChan, err := provider.GenerateCompletionStream(ctx, "gpt-3.5-turbo", input)
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
