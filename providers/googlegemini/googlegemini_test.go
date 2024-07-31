package googlegemini

import (
	"context"
	"os"
	"testing"

	"github.com/1broseidon/gollm/models"
)

func TestGoogleGeminiProvider(t *testing.T) {
	// Skip the test if GEMINI_API_KEY is not set
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping Google Gemini provider test")
	}

	ctx := context.Background()

	provider, err := NewGoogleGeminiProvider(ctx)
	if err != nil {
		t.Fatalf("Failed to create Google Gemini provider: %v", err)
	}
	defer provider.Close()

	t.Run("GenerateCompletion", func(t *testing.T) {
		input := models.CompletionInput{
			Messages: []models.ChatMessage{
				{Role: "user", Content: "Explain the concept of quantum computing in simple terms."},
			},
			MaxTokens:   100,
			Temperature: 0.7,
		}

		response, err := provider.GenerateCompletion(ctx, "gemini-1.5-pro", input)
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
			Messages: []models.ChatMessage{
				{Role: "user", Content: "Explain the concept of neural networks in simple terms."},
			},
			MaxTokens:   100,
			Temperature: 0.7,
			Stream:      true,
		}

		streamChan, err := provider.GenerateCompletionStream(ctx, "gemini-1.5-pro", input)
		if err != nil {
			t.Fatalf("GenerateCompletionStream failed: %v", err)
		}

		var fullText string
		for chunk := range streamChan {
			if chunk.Error != nil {
				t.Fatalf("Error in streaming: %v", chunk.Error)
			}
			fullText += chunk.Text
			if chunk.Done {
				break
			}
		}

		if fullText == "" {
			t.Error("Generated streaming text is empty")
		}
	})

	t.Run("SendChatMessage", func(t *testing.T) {
		session := provider.StartChat("gemini-1.5-pro")
		
		response, err := provider.SendChatMessage(ctx, session, "What is the capital of France?")
		if err != nil {
			t.Fatalf("SendChatMessage failed: %v", err)
		}

		if response.Text == "" {
			t.Error("Generated chat response is empty")
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
}
