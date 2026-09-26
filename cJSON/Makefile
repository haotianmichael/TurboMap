CFLAGS ?= -O3 -DNDEBUG -fPIC

all: cJSON.o

cJSON.o: cJSON.c cJSON.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f cJSON.o

.PHONY: all clean
