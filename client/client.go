package client

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/1broseidon/gollm/common"
	"github.com/1broseidon/gollm/internal/logging"
	"github.com/1broseidon/gollm/models"
	"github.com/1broseidon/gollm/providers/anthropic"
	"github.com/1broseidon/gollm/providers/googlegemini"
	"github.com/1broseidon/gollm/providers/ollama"
	"github.com/1broseidon/gollm/providers/openai"
)

// Provider interface defines the methods that each provider must implement
type Provider interface {
	GenerateCompletion(ctx context.Context, modelName string, input models.CompletionInput) (*models.CompletionResponse, error)
	GenerateCompletionStream(ctx context.Context, modelName string, input models.CompletionInput) (<-chan models.StreamingCompletionResponse, error)
	GenerateEmbedding(ctx context.Context, input string) ([]float32, error)
	StartChat(modelName string) interface{}
	SendChatMessage(ctx context.Context, session interface{}, message string) (*models.CompletionResponse, error)
	Close() error
}

// Client represents the main gollm client
type Client struct {
	providers       map[string]Provider
	defaultProvider string
	logger          logging.Logger
	mu              sync.RWMutex
}

// NewClient creates a new gollm client without automatic provider registration
func NewClient(ctx context.Context, options ...ClientOption) (*Client, error) {
	c := &Client{
		providers: make(map[string]Provider),
		logger:    logging.NewDefaultLogger(),
	}

	// Set default log level to Disabled
	c.logger.SetLevel(common.DisabledLevel)

	// Apply options
	for _, option := range options {
		option(c)
	}

	c.logger.Info("Initializing gollm client")

	return c, nil
}

func (c *Client) autoRegisterProviders(ctx context.Context) error {
	var wg sync.WaitGroup
	errChan := make(chan error, 4) // Buffer for potential errors from 4 providers

	wg.Add(4)

	go c.registerOpenAIProvider(&wg, errChan)
	go c.registerAnthropicProvider(&wg, errChan)
	go c.registerGoogleGeminiProvider(ctx, &wg, errChan)
	go c.registerOllamaProvider(&wg, errChan)

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			c.logger.Error("Error during provider registration:", err)
			// Non-blocking: log the error but continue with other providers
		}
	}

	return nil
}

func (c *Client) registerOpenAIProvider(wg *sync.WaitGroup, errChan chan<- error) {
	defer wg.Done()
	if openaiAPIKey := os.Getenv("OPENAI_API_KEY"); openaiAPIKey != "" {
		openaiProvider, err := openai.NewOpenAIProvider()
		if err != nil {
			errChan <- err
			return
		}
		c.RegisterProvider("openai", openaiProvider)
		c.setDefaultProviderIfEmpty("openai")
		c.logger.Info("Registered OpenAI provider")
	}
}

func (c *Client) registerAnthropicProvider(wg *sync.WaitGroup, errChan chan<- error) {
	defer wg.Done()
	if anthropicAPIKey := os.Getenv("ANTHROPIC_API_KEY"); anthropicAPIKey != "" {
		anthropicProvider, err := anthropic.NewAnthropicProvider()
		if err != nil {
			errChan <- err
			return
		}
		c.RegisterProvider("anthropic", anthropicProvider)
		c.setDefaultProviderIfEmpty("anthropic")
		c.logger.Info("Registered Anthropic provider")
	}
}

func (c *Client) registerGoogleGeminiProvider(ctx context.Context, wg *sync.WaitGroup, errChan chan<- error) {
	defer wg.Done()
	if geminiAPIKey := os.Getenv("GEMINI_API_KEY"); geminiAPIKey != "" {
		geminiProvider, err := googlegemini.NewGoogleGeminiProvider(ctx)
		if err != nil {
			errChan <- err
			return
		}
		c.RegisterProvider("googlegemini", geminiProvider)
		c.setDefaultProviderIfEmpty("googlegemini")
		c.logger.Info("Registered Google Gemini provider")
	}
}

func (c *Client) registerOllamaProvider(wg *sync.WaitGroup, errChan chan<- error) {
	defer wg.Done()
	if ollamaBaseURL := os.Getenv("OLLAMA_BASE_URL"); ollamaBaseURL != "" {
		ollamaProvider, err := ollama.NewOllamaProvider()
		if err != nil {
			errChan <- err
			return
		}
		c.RegisterProvider("ollama", ollamaProvider)
		c.setDefaultProviderIfEmpty("ollama")
		c.logger.Info("Registered Ollama provider")
	}
}

// setDefaultProviderIfEmpty sets the default provider if it hasn't been set yet
func (c *Client) setDefaultProviderIfEmpty(provider string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.defaultProvider == "" {
		c.defaultProvider = provider
	}
}

// RegisterProvider registers a new provider with the client
func (c *Client) RegisterProvider(name string, provider Provider) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.providers[name] = provider
}

// Close closes all provider clients
func (c *Client) Close() error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var wg sync.WaitGroup
	errChan := make(chan error, len(c.providers))

	for _, provider := range c.providers {
		wg.Add(1)
		go func(p Provider) {
			defer wg.Done()
			if err := p.Close(); err != nil {
				errChan <- err
			}
		}(provider)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	var lastErr error
	for err := range errChan {
		if err != nil {
			c.logger.Error("Error closing provider:", err)
			lastErr = err
		}
	}

	return lastErr
}

// GenerateCompletion generates a completion using the specified provider and model.
// It takes a context and a CompletionInput, which should include the provider/model
// in the format "provider/model" (e.g., "openai/gpt-3.5-turbo").
// The function returns a CompletionResponse or an error if the generation fails.
// GenerateCompletion generates a completion based on the provided input.
// It returns a CompletionResponse and any error encountered during the process.
func (c *Client) GenerateCompletion(ctx context.Context, input models.CompletionInput) (*models.CompletionResponse, error) {
	provider, model, err := c.parseProviderModel(input.Model)
	if err != nil {
		c.logger.Error("Failed to parse provider/model", "error", err)
		return nil, fmt.Errorf("failed to parse provider/model: %w", err)
	}

	c.mu.RLock()
	p, ok := c.providers[provider]
	c.mu.RUnlock()

	if !ok {
		// Provider not initialized, attempt to initialize it
		if err := c.initializeProvider(ctx, provider); err != nil {
			c.logger.Error("Failed to initialize provider:", provider, "error:", err)
			return nil, fmt.Errorf("failed to initialize provider %s: %w", provider, err)
		}

		c.mu.RLock()
		p, ok = c.providers[provider]
		c.mu.RUnlock()

		if !ok {
			c.logger.Error("Provider initialization failed:", provider)
			return nil, ErrUnsupportedProvider
		}
	}

	c.logger.Debugf("Generating completion with provider %s and model %s", provider, model)
	resp, err := p.GenerateCompletion(ctx, model, input)
	if err != nil {
		c.logger.Error("Failed to generate completion:", err)
		return nil, err
	}

	return resp, nil
}

// GenerateCompletionStream generates a streaming completion using the specified provider and model
func (c *Client) GenerateCompletionStream(ctx context.Context, input models.CompletionInput) (<-chan models.StreamingCompletionResponse, error) {
	provider, model, err := c.parseProviderModel(input.Model)
	if err != nil {
		c.logger.Error("Failed to parse provider/model:", err)
		return nil, err
	}

	c.mu.RLock()
	p, ok := c.providers[provider]
	c.mu.RUnlock()

	if !ok {
		c.logger.Error("Unsupported provider:", provider)
		return nil, ErrUnsupportedProvider
	}

	c.logger.Debugf("Generating streaming completion with provider %s and model %s", provider, model)
	stream, err := p.GenerateCompletionStream(ctx, model, input)
	if err != nil {
		c.logger.Error("Failed to generate streaming completion:", err)
		return nil, err
	}

	return stream, nil
}

// GenerateEmbedding generates an embedding using the default provider
func (c *Client) GenerateEmbedding(ctx context.Context, input string) ([]float32, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.defaultProvider == "" {
		c.logger.Error("No default provider set")
		return nil, errors.New("no default provider set")
	}

	provider, ok := c.providers[c.defaultProvider]
	if !ok {
		c.logger.Error("Unsupported default provider:", c.defaultProvider)
		return nil, ErrUnsupportedProvider
	}

	c.logger.Debugf("Generating embedding with default provider %s", c.defaultProvider)
	embedding, err := provider.GenerateEmbedding(ctx, input)
	if err != nil {
		c.logger.Error("Failed to generate embedding:", err)
		return nil, err
	}

	return embedding, nil
}

// StartChat starts a new chat session using the default provider
func (c *Client) StartChat() (interface{}, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.defaultProvider == "" {
		c.logger.Error("No default provider set")
		return nil, errors.New("no default provider set")
	}

	provider, ok := c.providers[c.defaultProvider]
	if !ok {
		c.logger.Error("Unsupported default provider:", c.defaultProvider)
		return nil, ErrUnsupportedProvider
	}

	c.logger.Debugf("Starting chat session with default provider %s", c.defaultProvider)
	session := provider.StartChat(c.defaultProvider)
	return session, nil
}

// SendChatMessage sends a message to an existing chat session using the default provider
func (c *Client) SendChatMessage(ctx context.Context, session interface{}, message string) (*models.CompletionResponse, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.defaultProvider == "" {
		c.logger.Error("No default provider set")
		return nil, errors.New("no default provider set")
	}

	provider, ok := c.providers[c.defaultProvider]
	if !ok {
		c.logger.Error("Unsupported default provider:", c.defaultProvider)
		return nil, ErrUnsupportedProvider
	}

	c.logger.Debugf("Sending chat message with default provider %s", c.defaultProvider)
	resp, err := provider.SendChatMessage(ctx, session, message)
	if err != nil {
		c.logger.Error("Failed to send chat message:", err)
		return nil, err
	}

	return resp, nil
}

// parseProviderModel splits the providerModel string into provider and model components.
// It returns an error if the string is not in the correct "provider/model" format.
func (c *Client) parseProviderModel(providerModel string) (string, string, error) {
	parts := strings.SplitN(providerModel, "/", 2)
	if len(parts) != 2 {
		return "", "", errors.New("invalid provider/model format")
	}
	return parts[0], parts[1], nil
}
// initializeProvider initializes a specific provider
func (c *Client) initializeProvider(ctx context.Context, providerName string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch providerName {
	case "openai":
		if openaiAPIKey := os.Getenv("OPENAI_API_KEY"); openaiAPIKey != "" {
			openaiProvider, err := openai.NewOpenAIProvider()
			if err != nil {
				return err
			}
			c.providers["openai"] = openaiProvider
			c.setDefaultProviderIfEmpty("openai")
			c.logger.Info("Registered OpenAI provider")
		} else {
			return errors.New("OPENAI_API_KEY not set")
		}
	case "anthropic":
		if anthropicAPIKey := os.Getenv("ANTHROPIC_API_KEY"); anthropicAPIKey != "" {
			anthropicProvider, err := anthropic.NewAnthropicProvider()
			if err != nil {
				return err
			}
			c.providers["anthropic"] = anthropicProvider
			c.setDefaultProviderIfEmpty("anthropic")
			c.logger.Info("Registered Anthropic provider")
		} else {
			return errors.New("ANTHROPIC_API_KEY not set")
		}
	case "googlegemini":
		if geminiAPIKey := os.Getenv("GEMINI_API_KEY"); geminiAPIKey != "" {
			geminiProvider, err := googlegemini.NewGoogleGeminiProvider(ctx)
			if err != nil {
				return err
			}
			c.providers["googlegemini"] = geminiProvider
			c.setDefaultProviderIfEmpty("googlegemini")
			c.logger.Info("Registered Google Gemini provider")
		} else {
			return errors.New("GEMINI_API_KEY not set")
		}
	case "ollama":
		if ollamaBaseURL := os.Getenv("OLLAMA_BASE_URL"); ollamaBaseURL != "" {
			ollamaProvider, err := ollama.NewOllamaProvider()
			if err != nil {
				return err
			}
			c.providers["ollama"] = ollamaProvider
			c.setDefaultProviderIfEmpty("ollama")
			c.logger.Info("Registered Ollama provider")
		} else {
			return errors.New("OLLAMA_BASE_URL not set")
		}
	default:
		return ErrUnsupportedProvider
	}

	return nil
}
