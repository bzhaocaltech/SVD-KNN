CXX = g++
CXXFLAGS = -std=c++14 -Wall -g

all: run_model run_knn

run_knn: run_knn.o knn/knn.o
	$(CXX) $(CXXFLAGS) $^ -o $@

run_model: run_model.o svd/svd.o
	$(CXX) $(CXXFLAGS) $^ -o run_model

run_model.o: run_model.cpp
	$(CXX) $(CXXFLAGS) -c run_model.cpp

clean:
	rm -f run_model *.o svd/*.o
