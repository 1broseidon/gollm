package googlegemini

import (
	"context"
	"errors"
	"os"
	"strings"

	"github.com/1broseidon/gollm/models"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// GoogleGeminiProvider implements the Google Gemini-specific functionality
type GoogleGeminiProvider struct {
	client *genai.Client
}

// NewGoogleGeminiProvider creates a new Google Gemini provider
func NewGoogleGeminiProvider(ctx context.Context) (*GoogleGeminiProvider, error) {
	apiKey, found := os.LookupEnv("GEMINI_API_KEY")
	if !found {
		return nil, errors.New("GEMINI_API_KEY environment variable is not set")
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}

	return &GoogleGeminiProvider{
		client: client,
	}, nil
}

// Close closes the Google Gemini client
func (p *GoogleGeminiProvider) Close() error {
	return p.client.Close()
}

// GenerateCompletion generates a completion using the specified Google Gemini model
func (p *GoogleGeminiProvider) GenerateCompletion(ctx context.Context, modelName string, input models.CompletionInput) (*models.CompletionResponse, error) {
	model := p.client.GenerativeModel(modelName)
	model.SetTemperature(float32(input.Temperature))
	p.SetMaxOutputTokens(model, input.MaxTokens)

	prompt := genai.Text(input.Messages[len(input.Messages)-1].Content)
	resp, err := model.GenerateContent(ctx, prompt)
	if err != nil {
		return nil, err
	}

	if len(resp.Candidates) == 0 {
		return nil, errors.New("no content generated")
	}

	generatedText, ok := resp.Candidates[0].Content.Parts[0].(genai.Text)
	if !ok {
		return nil, errors.New("unexpected content type in response")
	}

	generatedString := string(generatedText)

	inputTokenCount, err := p.CountTokens(ctx, modelName, input.Messages[len(input.Messages)-1].Content)
	if err != nil {
		return nil, err
	}

	outputTokenCount, err := p.CountTokens(ctx, modelName, generatedString)
	if err != nil {
		return nil, err
	}

	return &models.CompletionResponse{
		Text: generatedString,
		Usage: &models.Usage{
			PromptTokens:     inputTokenCount,
			CompletionTokens: outputTokenCount,
			TotalTokens:      inputTokenCount + outputTokenCount,
		},
	}, nil
}

// GenerateCompletionStream generates a streaming completion using the specified Google Gemini model
func (p *GoogleGeminiProvider) GenerateCompletionStream(ctx context.Context, modelName string, input models.CompletionInput) (<-chan models.StreamingCompletionResponse, error) {
	model := p.client.GenerativeModel(modelName)
	model.SetTemperature(float32(input.Temperature))
	p.SetMaxOutputTokens(model, input.MaxTokens)

	prompt := genai.Text(input.Messages[len(input.Messages)-1].Content)
	iter := model.GenerateContentStream(ctx, prompt)

	streamChan := make(chan models.StreamingCompletionResponse)

	go func() {
		defer close(streamChan)

		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				streamChan <- models.StreamingCompletionResponse{Done: true}
				return
			}
			if err != nil {
				streamChan <- models.StreamingCompletionResponse{Error: err}
				return
			}

			if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
				continue
			}

			text, ok := resp.Candidates[0].Content.Parts[0].(genai.Text)
			if !ok {
				streamChan <- models.StreamingCompletionResponse{Error: errors.New("unexpected content type in response")}
				return
			}

			streamChan <- models.StreamingCompletionResponse{
				Text: string(text),
			}
		}
	}()

	return streamChan, nil
}

// CountTokens counts the number of tokens in the given content
func (p *GoogleGeminiProvider) CountTokens(ctx context.Context, modelName string, content string) (int, error) {
	model := p.client.GenerativeModel(modelName)
	cs := model.StartChat()
	resp, err := cs.SendMessage(ctx, genai.Text(content))
	if err != nil {
		return 0, err
	}

	// Use the total token count from the response
	if resp.Candidates != nil && len(resp.Candidates) > 0 {
		return int(resp.Candidates[0].TokenCount), nil
	}

	// If token count is not available, return an error
	return 0, errors.New("token count information not available")
}

// SetMaxOutputTokens sets the max output tokens for the model
func (p *GoogleGeminiProvider) SetMaxOutputTokens(model *genai.GenerativeModel, maxTokens int) {
	if maxTokens > 0 {
		model.SetMaxOutputTokens(int32(maxTokens))
	}
}

// GenerateEmbedding generates an embedding using the Google Gemini model
func (p *GoogleGeminiProvider) GenerateEmbedding(ctx context.Context, input string) ([]float32, error) {
	// TODO: Implement embedding generation
	return nil, errors.New("embedding generation not implemented")
}

// StartChat starts a new chat session
func (p *GoogleGeminiProvider) StartChat(modelName string) interface{} {
	model := p.client.GenerativeModel(modelName)
	return model.StartChat()
}

// SendChatMessage sends a message to an existing chat session
func (p *GoogleGeminiProvider) SendChatMessage(ctx context.Context, session interface{}, message string) (*models.CompletionResponse, error) {
	chatSession, ok := session.(*genai.ChatSession)
	if !ok {
		return nil, errors.New("invalid chat session type")
	}
	resp, err := chatSession.SendMessage(ctx, genai.Text(message))
	if err != nil {
		return nil, err
	}

	if len(resp.Candidates) == 0 {
		return nil, errors.New("no content generated")
	}

	generatedText := ""
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			generatedText += string(text)
		}
	}

	promptTokens := resp.Candidates[0].TokenCount
	completionTokens := int32(len(strings.Split(generatedText, " "))) // Rough estimate

	return &models.CompletionResponse{
		Text: generatedText,
		Usage: &models.Usage{
			PromptTokens:     int(promptTokens),
			CompletionTokens: int(completionTokens),
			TotalTokens:      int(promptTokens + completionTokens),
		},
	}, nil
}
