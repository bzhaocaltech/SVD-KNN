CXX = g++
CXXFLAGS = -std=c++14 -Wall -g

all: run_model

run_model: run_model.o
	$(CXX) $(CXXFLAGS) run_model.o -o run_model

run_model.o: run_model.cpp
	$(CXX) $(CXXFLAGS) -c run_model.cpp

clean:
	rm -f run_model *.o
