package common

// LogLevel represents the logging level
type LogLevel int

const (
	// DisabledLevel disables all logging
	DisabledLevel LogLevel = iota
	// DebugLevel sets the logging level to debug
	DebugLevel
	// InfoLevel sets the logging level to info
	InfoLevel
	// WarnLevel sets the logging level to warn
	WarnLevel
	// ErrorLevel sets the logging level to error
	ErrorLevel
)
