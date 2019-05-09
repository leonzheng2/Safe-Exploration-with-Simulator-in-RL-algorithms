all: 	
	make -f Makefile.environment 
	make -f Makefile.experiment

clean:
	make -f Makefile.environment clean
	make -f Makefile.experiment clean
