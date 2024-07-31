package client

import (
	"errors"
	"github.com/1broseidon/gollm/common"
	"github.com/1broseidon/gollm/internal/logging"
)

// ErrUnsupportedProvider is returned when an unsupported provider is specified
var ErrUnsupportedProvider = errors.New("unsupported provider")

// ClientOption is a function type for configuring the Client.
// It allows for flexible and extensible client configuration.
type ClientOption func(*Client)

// WithDefaultProvider sets the default provider for the client.
// This provider will be used when no specific provider is specified for operations.
// If not set, the client will attempt to use the first registered provider as the default.
func WithDefaultProvider(provider string) ClientOption {
	return func(c *Client) {
		c.defaultProvider = provider
	}
}

// WithLogger sets the logger for the client.
// The provided logger will be used for all logging operations within the client.
func WithLogger(logger logging.Logger) ClientOption {
	return func(c *Client) {
		c.logger = logger
	}
}

// WithLogLevel sets the log level for the client.
// This option will only take effect if the client's logger supports setting log levels.
func WithLogLevel(level common.LogLevel) ClientOption {
	return func(c *Client) {
		if logger, ok := c.logger.(interface{ SetLevel(common.LogLevel) }); ok {
			logger.SetLevel(level)
		}
	}
}
