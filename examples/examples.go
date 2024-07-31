package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/1broseidon/gollm/client"
	"github.com/1broseidon/gollm/common"
	"github.com/1broseidon/gollm/models"
)

func main() {
	ctx := context.Background()

	if len(os.Args) < 2 {
		showAvailableModels()
		return
	}

	fmt.Printf("Running example for: %s\n", os.Args[1])

	switch os.Args[1] {
	case "logging":
		c, err := client.NewClient(ctx, client.WithLogLevel(common.InfoLevel))
		if err != nil {
			log.Fatalf("Failed to create gollm client: %v", err)
		}
		defer c.Close()
		fmt.Println("Logging enabled. Running all examples with logging:")
		openAIExample(ctx, c)
		geminiExample(ctx, c)
		anthropicExample(ctx, c)
		ollamaExample(ctx, c)
	case "openai":
		c, err := client.NewClient(ctx, client.WithDefaultProvider("openai"), client.WithLogLevel(common.InfoLevel))
		if err != nil {
			log.Fatalf("Failed to create gollm client: %v", err)
		}
		defer c.Close()
		openAIExample(ctx, c)
	case "gemini":
		c, err := client.NewClient(ctx, client.WithDefaultProvider("googlegemini"), client.WithLogLevel(common.InfoLevel))
		if err != nil {
			log.Fatalf("Failed to create gollm client: %v", err)
		}
		defer c.Close()
		geminiExample(ctx, c)
	case "anthropic":
		c, err := client.NewClient(ctx, client.WithDefaultProvider("anthropic"), client.WithLogLevel(common.InfoLevel))
		if err != nil {
			log.Fatalf("Failed to create gollm client: %v", err)
		}
		defer c.Close()
		fmt.Println("Client created successfully")
		anthropicExample(ctx, c)
	case "ollama":
		c, err := client.NewClient(ctx, client.WithDefaultProvider("ollama"), client.WithLogLevel(common.InfoLevel))
		if err != nil {
			log.Fatalf("Failed to create gollm client: %v", err)
		}
		defer c.Close()
		ollamaExample(ctx, c)
	case "all":
		c, err := client.NewClient(ctx, client.WithLogLevel(common.InfoLevel))
		if err != nil {
			log.Fatalf("Failed to create gollm client: %v", err)
		}
		defer c.Close()
		openAIExample(ctx, c)
		geminiExample(ctx, c)
		anthropicExample(ctx, c)
		ollamaExample(ctx, c)
	default:
		fmt.Println("Invalid provider. Available options: logging, openai, gemini, anthropic, ollama, all")
	}
}

func showAvailableModels() {
	fmt.Println("Available options:")
	fmt.Println("- logging (runs all examples with logging enabled)")
	fmt.Println("- openai")
	fmt.Println("- gemini")
	fmt.Println("- anthropic")
	fmt.Println("- ollama")
	fmt.Println("- all (runs all examples)")
	fmt.Println("\nUsage: go run examples.go <option>")
}

func openAIExample(ctx context.Context, c *client.Client) {
	fmt.Println("Starting OpenAI example")
	openAIInput := models.CompletionInput{
		Model: "openai/gpt-3.5-turbo",
		Messages: []models.ChatMessage{
			{Role: "user", Content: "Tell me a short story about a robot and a human. Max 50 words."},
		},
		MaxTokens:   200,
		Temperature: 0.7,
		Stream:      true,
	}
	fmt.Println("Calling GenerateCompletionStream")
	streamChan, err := c.GenerateCompletionStream(ctx, openAIInput)
	if err != nil {
		log.Printf("Failed to generate completion stream with OpenAI: %v", err)
		return
	}
	fmt.Println("Successfully got stream channel")

	fmt.Println("OpenAI GPT-3.5-turbo Response:")
	var openAIResponseText string
	var openAIUsage *models.Usage

	// Create a timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	done := make(chan bool)
	go func() {
		for {
			select {
			case chunk, ok := <-streamChan:
				if !ok {
					done <- true
					return
				}
				if chunk.Error != nil {
					log.Printf("Error in streaming: %v", chunk.Error)
					done <- true
					return
				}
				fmt.Print(chunk.Text)
				openAIResponseText += chunk.Text
				if chunk.Usage != nil {
					openAIUsage = chunk.Usage
				}
				if chunk.Done {
					done <- true
					return
				}
			case <-timeoutCtx.Done():
				log.Println("Timeout occurred while waiting for OpenAI response")
				done <- true
				return
			}
		}
	}()

	<-done

	if openAIResponseText == "" {
		log.Println("No response text received from OpenAI")
	} else {
		fmt.Println("")
	}

	if openAIUsage != nil {
		fmt.Printf("\nToken Usage:\nInput Tokens: %d\nOutput Tokens: %d\nTotal Tokens: %d\n",
			openAIUsage.PromptTokens, openAIUsage.CompletionTokens, openAIUsage.TotalTokens)
	} else {
		fmt.Println("Token Usage information is missing")
	}
}

func geminiExample(ctx context.Context, c *client.Client) {
	geminiInput := models.CompletionInput{
		Model: "googlegemini/gemini-1.5-flash",
		Messages: []models.ChatMessage{
			{Role: "user", Content: "Briefly explain the concept of machine learning. Max 50 words."},
		},
		MaxTokens:   200,
		Temperature: 0.7,
	}

	geminiInput.Stream = true
	streamChan, err := c.GenerateCompletionStream(ctx, geminiInput)
	if err != nil {
		log.Printf("Failed to generate completion stream with Google Gemini: %v", err)
		return
	}

	fmt.Println("\nGoogle Gemini Response:")
	var geminiResponseText string
	var geminiUsage *models.Usage

	for chunk := range streamChan {
		if chunk.Error != nil {
			log.Printf("Error in streaming: %v", chunk.Error)
			return
		}
		fmt.Print(chunk.Text)
		geminiResponseText += chunk.Text
		if chunk.Usage != nil {
			geminiUsage = chunk.Usage
		}
		if chunk.Done {
			break
		}
	}

	if geminiUsage != nil {
		fmt.Printf("\n\nToken Usage:\nInput Tokens: %d\nOutput Tokens: %d\nTotal Tokens: %d\n",
			geminiUsage.PromptTokens, geminiUsage.CompletionTokens, geminiUsage.TotalTokens)
	} else {
		// Calculate approximate token usage
		inputTokens := len(strings.Split(geminiInput.Messages[0].Content, " "))
		outputTokens := len(strings.Split(geminiResponseText, " "))
		totalTokens := inputTokens + outputTokens

		fmt.Printf("\n\nApproximate Token Usage:\nInput Tokens: %d\nOutput Tokens: %d\nTotal Tokens: %d\n",
			inputTokens, outputTokens, totalTokens)
	}
}

func anthropicExample(ctx context.Context, c *client.Client) {
	fmt.Println("Starting Anthropic example")
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Println("\nSkipping Anthropic example: ANTHROPIC_API_KEY not set")
		return
	}

	anthropicInput := models.CompletionInput{
		Model:    "anthropic/claude-3-haiku-20240307",
		Provider: "anthropic",
		Messages: []models.ChatMessage{
			{Role: "user", Content: "Explain the concept of quantum computing in simple terms. Max 50 words."},
		},
		MaxTokens:   200,
		Temperature: 0.7,
	}

	fmt.Println("Calling GenerateCompletionStream")
	anthropicInput.Stream = true
	fmt.Println("Debug: About to call c.GenerateCompletionStream")
	streamChan, err := c.GenerateCompletionStream(ctx, anthropicInput)
	if err != nil {
		log.Printf("Failed to generate completion stream with Anthropic: %v", err)
		return
	}
	fmt.Println("Successfully got stream channel")
	fmt.Println("Debug: Stream channel received, about to start processing")

	fmt.Println("\nAnthropic Claude Response:")
	var anthropicResponseText string
	var anthropicUsage *models.Usage

	// Create a timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	done := make(chan bool)
	go func() {
		for {
			select {
			case chunk, ok := <-streamChan:
				if !ok {
					fmt.Println("Stream channel closed")
					done <- true
					return
				}
				fmt.Printf("Debug: Received chunk: %+v\n", chunk)
				if chunk.Error != nil {
					log.Printf("Error in streaming: %v", chunk.Error)
					done <- true
					return
				}
				fmt.Print(chunk.Text)
				anthropicResponseText += chunk.Text
				if chunk.Usage != nil {
					anthropicUsage = chunk.Usage
				}
				if chunk.Done {
					fmt.Println("Stream completed")
					done <- true
					return
				}
			case <-timeoutCtx.Done():
				log.Println("Timeout occurred while waiting for Anthropic response")
				done <- true
				return
			}
		}
	}()

	select {
	case <-done:
		fmt.Println("Stream processing completed")
	case <-timeoutCtx.Done():
		log.Println("Timeout occurred while waiting for stream to complete")
	}

	if anthropicResponseText == "" {
		log.Println("No response text received from Anthropic")
	} else {
		fmt.Println("Response received successfully")
	}

	if anthropicUsage != nil {
		fmt.Printf("\nToken Usage:\nInput Tokens: %d\nOutput Tokens: %d\nTotal Tokens: %d\n",
			anthropicUsage.PromptTokens, anthropicUsage.CompletionTokens, anthropicUsage.TotalTokens)
	} else {
		fmt.Println("Token Usage information is missing")
	}

	fmt.Println("Anthropic example completed")
}
func ollamaExample(ctx context.Context, c *client.Client) {
	if os.Getenv("OLLAMA_BASE_URL") == "" {
		fmt.Println("\nSkipping Ollama example: OLLAMA_BASE_URL not set")
		return
	}

	ollamaInput := models.CompletionInput{
		Model: "ollama/llama3.1:latest",
		Messages: []models.ChatMessage{
			{Role: "user", Content: "Explain the concept of quantum entanglement in simple terms. Max 50 words."},
		},
		MaxTokens:   200,
		Temperature: 0.7,
	}

	ollamaInput.Stream = true
	streamChan, err := c.GenerateCompletionStream(ctx, ollamaInput)
	if err != nil {
		log.Printf("Failed to generate completion stream with Ollama: %v", err)
		return
	}

	fmt.Println("\nOllama Llama3.1 Response:")
	var ollamaResponseText string
	var ollamaUsage *models.Usage

	for chunk := range streamChan {
		if chunk.Error != nil {
			log.Printf("Error in streaming: %v", chunk.Error)
			return
		}
		fmt.Print(chunk.Text)
		ollamaResponseText += chunk.Text
		if chunk.Usage != nil {
			ollamaUsage = chunk.Usage
		}
		if chunk.Done {
			break
		}
	}

	if ollamaUsage != nil {
		fmt.Printf("\n\nToken Usage:\nInput Tokens: %d\nOutput Tokens: %d\nTotal Tokens: %d\n",
			ollamaUsage.PromptTokens, ollamaUsage.CompletionTokens, ollamaUsage.TotalTokens)
	} else {
		fmt.Println("Token Usage information is missing")
	}
}
