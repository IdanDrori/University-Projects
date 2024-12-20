CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

build-python:
	python3 setup.py build_ext --inplace

build-c: symnmf.o symnmf.h
	gcc -o symnmf symnmf.o -lm

symnmf.o: symnmf.c
	gcc -c symnmf.c $(CFLAGS)

