package models

// CompletionInput represents the input for a completion request.
type CompletionInput struct {
	Model       string
	Messages    []ChatMessage
	MaxTokens   int
	Temperature float32
	Stream      bool
	Provider    string // Specifies the provider explicitly
}

// ChatMessage represents a message in a chat conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionResponse represents the response from a completion request.
type CompletionResponse struct {
	Text     string
	Usage    *Usage
	Provider string // Indicates which provider generated the response
}

// Usage represents the token usage information for a completion request.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// StreamingCompletionResponse represents a chunk of a streaming completion response.
type StreamingCompletionResponse struct {
	Text     string
	Done     bool
	Error    error
	Usage    *Usage
	Provider string // Indicates which provider generated the response
}

// ProviderOptions represents additional options specific to each provider.
type ProviderOptions struct {
	OpenAI       OpenAIOptions
	GoogleGemini GoogleGeminiOptions
	Anthropic    AnthropicOptions
	Ollama       OllamaOptions
}

// OpenAIOptions represents OpenAI-specific options.
type OpenAIOptions struct {
	// TODO: Add OpenAI-specific fields here
}

// GoogleGeminiOptions represents Google Gemini-specific options.
type GoogleGeminiOptions struct {
	// TODO: Add Google Gemini-specific fields here
}

// AnthropicOptions represents Anthropic-specific options.
type AnthropicOptions struct {
	// TODO: Add Anthropic-specific fields here
}

// OllamaOptions represents Ollama-specific options.
type OllamaOptions struct {
	// TODO: Add Ollama-specific fields here
}
