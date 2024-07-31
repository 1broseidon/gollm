package logging

import (
	"log"
	"os"
	"sync"

	"github.com/1broseidon/gollm/common"
)

type Logger interface {
	Debug(args ...interface{})
	Debugf(format string, args ...interface{})
	Info(args ...interface{})
	Infof(format string, args ...interface{})
	Warn(args ...interface{})
	Warnf(format string, args ...interface{})
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	SetLevel(level common.LogLevel)
}

type defaultLogger struct {
	logger *log.Logger
	level  common.LogLevel
	mu     sync.Mutex
}

func NewDefaultLogger() Logger {
	return &defaultLogger{
		logger: log.New(os.Stderr, "", log.LstdFlags),
		level:  common.DisabledLevel,
	}
}

func (l *defaultLogger) log(level common.LogLevel, prefix string, args ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if level >= l.level {
		l.logger.Print(append([]interface{}{prefix}, args...)...)
	}
}

func (l *defaultLogger) logf(level common.LogLevel, prefix, format string, args ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if level >= l.level {
		l.logger.Printf(prefix+format, args...)
	}
}

func (l *defaultLogger) Debug(args ...interface{}) { l.log(common.DebugLevel, "DEBUG: ", args...) }
func (l *defaultLogger) Debugf(format string, args ...interface{}) {
	l.logf(common.DebugLevel, "DEBUG: ", format, args...)
}
func (l *defaultLogger) Info(args ...interface{}) { l.log(common.InfoLevel, "INFO: ", args...) }
func (l *defaultLogger) Infof(format string, args ...interface{}) {
	l.logf(common.InfoLevel, "INFO: ", format, args...)
}
func (l *defaultLogger) Warn(args ...interface{}) { l.log(common.WarnLevel, "WARN: ", args...) }
func (l *defaultLogger) Warnf(format string, args ...interface{}) {
	l.logf(common.WarnLevel, "WARN: ", format, args...)
}
func (l *defaultLogger) Error(args ...interface{}) { l.log(common.ErrorLevel, "ERROR: ", args...) }
func (l *defaultLogger) Errorf(format string, args ...interface{}) {
	l.logf(common.ErrorLevel, "ERROR: ", format, args...)
}

func (l *defaultLogger) SetLevel(level common.LogLevel) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = level
}
