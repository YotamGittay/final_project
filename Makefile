CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
TARGET = symnmf

all: $(TARGET)

$(TARGET): symnmf.c
	$(CC) $(CFLAGS) -o $(TARGET) symnmf.c -lm

clean:
	rm -f $(TARGET)
