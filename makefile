CC = g++-10
CCFLAGS = -std=c++2a -O3
LINKFLAGS = -lstdc++fs -lboost_timer

all: main

main.o: main.cc pandas.h parsers.h frame.h series.h
	$(CC) $(CCFLAGS) -c $< -o $@

main: main.o
	$(CC) $(CCFLAGS) $^ $(LINKFLAGS) -o $@

clean:
	rm -rf *.o

clobber: clean
	rm -rf main
