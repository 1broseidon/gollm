package models

// CompletionInput represents the input for a completion request
type CompletionInput struct {
	Model       string
	Messages    []ChatMessage
	MaxTokens   int
	Temperature float32
	Stream      bool
	Provider    string // Added to specify the provider explicitly
}

// ChatMessage represents a message in a chat conversation
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionResponse represents the response from a completion request
type CompletionResponse struct {
	Text    string
	Usage   *Usage
	Provider string // Added to indicate which provider generated the response
}

// Usage represents the token usage information
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// StreamingCompletionResponse represents a chunk of a streaming completion response
type StreamingCompletionResponse struct {
	Text     string
	Done     bool
	Error    error
	Usage    *Usage
	Provider string // Added to indicate which provider generated the response
}

// ProviderOptions represents additional options specific to each provider
type ProviderOptions struct {
	OpenAI       OpenAIOptions
	GoogleGemini GoogleGeminiOptions
	Anthropic    AnthropicOptions
}

// OpenAIOptions represents OpenAI-specific options
type OpenAIOptions struct {
	// Add OpenAI-specific fields here
}

// GoogleGeminiOptions represents Google Gemini-specific options
type GoogleGeminiOptions struct {
	// Add Google Gemini-specific fields here
}

// AnthropicOptions represents Anthropic-specific options
type AnthropicOptions struct {
	// Add Anthropic-specific fields here
}
