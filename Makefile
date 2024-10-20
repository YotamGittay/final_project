# Variables
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
TARGET = symnmf
OBJS = symnmf.o symnmf_helpers.o  # Object files

# Default target
all: $(TARGET)

# Linking the object files to create the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

# Rule to compile symnmf.c to symnmf.o
symnmf.o: symnmf.c symnmf_helpers.h
	$(CC) $(CFLAGS) -c symnmf.c

# Rule to compile symnmf_helpers.c to symnmf_helpers.o
symnmf_helpers.o: symnmf_helpers.c symnmf_helpers.h
	$(CC) $(CFLAGS) -c symnmf_helpers.c

# Clean target to remove generated files
clean:
	rm -f $(TARGET) $(OBJS)
