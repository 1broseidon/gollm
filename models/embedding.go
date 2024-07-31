package models

// EmbeddingInput represents the input for an embedding request.
type EmbeddingInput struct {
	Model    string
	Text     string
	Provider string
}

// EmbeddingResponse represents the response from an embedding request.
type EmbeddingResponse struct {
	Embedding []float32
	Usage     *Usage
	Provider  string
}
