LDFLAGS :=  -lm -lrlutils -lrlenvironment -lrlgluenetdev
CFLAGS :=   -Wall -pedantic

all: SwimmerEnvironment

SwimmerEnvironment: SwimmerEnvironment.o 
	$(CXX)   SwimmerEnvironment.o $(LDFLAGS) -o SwimmerEnvironment      

SwimmerEnvironment.o: SwimmerEnvironment.cpp
	$(CXX)  $(CFLAGS) -c SwimmerEnvironment.cpp -o SwimmerEnvironment.o 

clean:
	rm -rf SwimmerEnvironment SwimmerEnvironment.exe *.o