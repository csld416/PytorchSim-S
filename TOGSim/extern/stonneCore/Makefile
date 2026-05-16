# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O0 -g -fPIC -Iinclude/ -Iexternal/
LDFLAGS = -shared

# Directories
SRC_DIR = src
INCLUDE_DIR = include
EXTERNAL_DIR = external
OBJSDIR = objs
LIB_DIR = lib

# Source and header files
SOURCE =   $(wildcard $(SRC_DIR)/*.cpp)

INCLUDES = $(wildcard $(INCLUDE_DIR)/*.h)


OBJS = $(patsubst $(SRC_DIR)/%, $(OBJSDIR)/%, $(patsubst %.cpp,%.o,$(SOURCE)))

# Output libraries
STATIC_LIB = $(LIB_DIR)/libsstStonne.a
SHARED_LIB = $(LIB_DIR)/libsstStonne.so

all: $(STATIC_LIB) $(SHARED_LIB)

# Static library build
$(STATIC_LIB): $(OBJSDIR) $(OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $(OBJS)

# Shared library build
$(SHARED_LIB): $(OBJSDIR) $(OBJS)
	@mkdir -p $(LIB_DIR)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS)

# Create object directory
$(OBJSDIR):
	mkdir -p $@ && \
	mkdir -p $@/TileGenerator && \
	mkdir -p $@/TileGenerator/mRNA && \
	mkdir -p $@/TileGenerator/StonneMapper && \
	mkdir -p $@/TileGenerator/Utils

# Compile source files into object files
$(OBJSDIR)/%.o: $(SRC_DIR)/%.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Install the library and headers
install:
	@mkdir -p /usr/local/include/sstStonne
	@mkdir -p /usr/local/lib
	cp $(STATIC_LIB) /usr/local/lib/
	cp $(SHARED_LIB) /usr/local/lib/
	cp $(INCLUDES) /usr/local/include/sstStonne/

# Clean build files
.PHONY: clean
clean:
	rm -rf $(OBJSDIR) $(LIB_DIR):w

