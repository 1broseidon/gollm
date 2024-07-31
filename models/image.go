package models

// ImageGenerationInput represents the input for an image generation request.
type ImageGenerationInput struct {
	Prompt   string
	Size     string
	Number   int
	Provider string
}

// ImageGenerationResponse represents the response from an image generation request.
type ImageGenerationResponse struct {
	Images   []string // URLs or base64-encoded image data
	Usage    *Usage
	Provider string
}
