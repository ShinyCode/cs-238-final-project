CC = gcc
CFLAGS = -std=c99 -g -Wno-deprecated-declarations

default: malloc_count

malloc_count.o: malloc_count.c
	$(CC) $(CFLAGS) -c malloc_count.c -o malloc_count.o

malloc_count: malloc_count.o
	$(CC) $(CFLAGS) malloc_count.o -o malloc_count

clean:
	-rm -f malloc_count.o
	-rm -f malloc_count
