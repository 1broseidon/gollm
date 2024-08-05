# gollm: A Unified Go Interface for Large Language Models

gollm is a Go library designed to simplify interactions with various Large Language Models (LLMs) from different providers. It provides a consistent interface for calling completion, embedding, and image generation endpoints, regardless of the underlying provider.

## Installation

To use gollm in your Go project, install it using `go get`:

```bash
go get github.com/1broseidon/gollm
```

## Usage

### Client Initialization

To use gollm, you first need to initialize a client. The client will automatically register providers based on the environment variables set for API keys.

```go
import (
    "context"
    "github.com/1broseidon/gollm/client"
)

func main() {
    ctx := context.Background()

    // Create a new client with default settings
    c, err := client.NewClient(ctx)
    if err != nil {
        // Handle error
    }
    defer c.Close()

    // Use the client...
}
```

### Streaming Completion Example

Here's an example of how to use the client to stream a completion from a specific provider and model:

```go
import (
    "context"
    "fmt"
    "log"

    "github.com/1broseidon/gollm/client"
    "github.com/1broseidon/gollm/models"
)

func main() {
    ctx := context.Background()

    c, err := client.NewClient(ctx)
    if err != nil {
        log.Fatalf("Failed to create gollm client: %v", err)
    }
    defer c.Close()

    input := models.CompletionInput{
        Model: "openai/gpt-3.5-turbo", // Specify provider and model
        Messages: []models.ChatMessage{
            {Role: "user", Content: "Tell me a short story about a robot and a human."},
        },
        MaxTokens:   200,
        Temperature: 0.7,
        Stream:      true,
    }

    streamChan, err := c.GenerateCompletionStream(ctx, input)
    if err != nil {
        log.Fatalf("Failed to generate completion stream: %v", err)
    }

    for chunk := range streamChan {
        if chunk.Error != nil {
            log.Printf("Error in streaming: %v", chunk.Error)
            return
        }
        fmt.Print(chunk.Text)
        if chunk.Done {
            break
        }
    }
}
```

This example demonstrates how to:

1. Initialize the gollm client
2. Create a completion input with a specific provider and model
3. Generate a streaming completion
4. Process the streaming response
5. Enjoy!

Remember to set the appropriate environment variables for the API keys of the providers you want to use. For example:

```bash
export OPENAI_API_KEY=your_openai_api_key
export GEMINI_API_KEY=your_gemini_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export OLLAMA_BASE_URL=http://localhost:11434
```

## Supported Providers

gollm currently supports the following providers:

- OpenAI
- Google Gemini
- Anthropic
- Ollama

Each provider requires its own API key or base URL to be set as an environment variable.

## Contributing

Contributions to gollm are welcome! Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## License

gollm is released under the MIT License. See the LICENSE file for more details.
