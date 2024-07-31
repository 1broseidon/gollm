package common

// LogLevel represents the logging level
type LogLevel int

const (
	// DisabledLevel disables all logging. Use this to turn off logging completely.
	DisabledLevel LogLevel = iota

	// DebugLevel sets the logging level to debug. This level is used for detailed system operations.
	DebugLevel

	// InfoLevel sets the logging level to info. Use this for general operational entries about what's happening inside the application.
	InfoLevel

	// WarnLevel sets the logging level to warn. This level is used for non-critical entries that deserve eyes.
	WarnLevel

	// ErrorLevel sets the logging level to error. This level is used for errors that should definitely be noted and investigated.
	ErrorLevel
)
