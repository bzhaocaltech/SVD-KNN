CXX = g++
CXXFLAGS = -std=c++14 -Wall -g

all: svd.o

svd.o: svd.cpp svd.hpp
	$(CXX) $(CXXFLAGS) -c svd.cpp

clean:
	rm -f *.o
