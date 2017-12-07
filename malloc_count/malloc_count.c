// Malloc hook information taken from https://www.gnu.org/software/libc/manual/html_node/Hooks-for-Malloc.html

#include <stdlib.h>
#include <malloc.h>

static void *(*old_malloc_hook)(size_t, const void *);
static void (*old_free_hook)(void *, const void *);

static void my_init_hook(void);
static void *my_malloc_hook(size_t, const void *);
static void my_free_hook(void *, const void *);

static void on_malloc(size_t, const void *);
static void on_free(void *, const void*);

FILE *out;

static void my_init(void) {
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
}

static void *my_malloc_hook(size_t size, const void *caller) {
  void *result;
  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  result = malloc(size);
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  on_malloc(size, caller);
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
  return result;
}

static void my_free_hook(void *ptr, const void *caller) {
  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  free(ptr);
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  on_free(ptr, caller);
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
}

static void on_malloc(size_t size, const void *caller) {
  // printf("M %zu\n", size);
  fprintf(out, "M %zu\n", size);
}

static void on_free(void *ptr, const void *caller) {

}

static void run() {
  // Do stuff here that uses malloc  
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: ./alloc_hook [outfile]\n");
    return -1;
  }
  char *outfile = argv[1];
  out = fopen(outfile, "w+");
  if (out == NULL) {
    printf("Could not open the file for output!\n");
    return -1;
  }
  my_init();
  run();
  fclose(out);
  return 0;
}
