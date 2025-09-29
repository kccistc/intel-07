CC      := gcc
CFLAGS  := -O2 -Wall -Wextra -Iinclude
LDFLAGS := -lpthread -lmysqlclient

SRCS = \
  src/main.c \
  src/util.c \
  src/db.c \
  src/uart.c \
  src/server.c

OBJS = $(SRCS:.c=.o)

server: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJS) server