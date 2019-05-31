CXX = g++
CXXFLAGS = -std=c++14 -Wall -g

all: run_svd run_knn

run_knn: run_knn.o knn/knn.o
	$(CXX) $(CXXFLAGS) $^ -o $@

run_svd: run_svd.o svd/svd.o
	$(CXX) $(CXXFLAGS) $^ -o run_svd

clean:
	rm -f run_svd run_knn *.o svd/*.o knn/*.o
