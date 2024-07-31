package client

import (
	"errors"
	"github.com/1broseidon/gollm/common"
	"github.com/1broseidon/gollm/internal/logging"
)

// ErrUnsupportedProvider is returned when an unsupported provider is specified
var ErrUnsupportedProvider = errors.New("unsupported provider")

// ClientOption is a function type for configuring the client
type ClientOption func(*Client)

// WithDefaultProvider sets the default provider for the client
func WithDefaultProvider(provider string) ClientOption {
	return func(c *Client) {
		c.defaultProvider = provider
	}
}

// WithLogger sets the logger for the client
func WithLogger(logger logging.Logger) ClientOption {
	return func(c *Client) {
		c.logger = logger
	}
}

// WithLogLevel sets the log level for the client
func WithLogLevel(level common.LogLevel) ClientOption {
	return func(c *Client) {
		if logger, ok := c.logger.(interface{ SetLevel(common.LogLevel) }); ok {
			logger.SetLevel(level)
		}
	}
}
