# Makefie for Bifrost-STONNE API
PKG_CFLAGS = -std=c++17 -O2 -fPIC -Iinclude/ -Iexternal/\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I$(STONNE_ROOT)/include\
	-I$(STONNE_ROOT)/external\
        -I$(MRNA_ROOT)/src\

PKG_LDFLAGS =-L${TVM_ROOT}/build 

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	PKG_LDFLAGS += -stdlib=libc++ -undefined dynamic_lookup -DCMAKE_FIND_FRAMEWORK=LAST
endif
CC=clang
../bifrost/bifrost/stonne/stonne_lib/stonne_lib.so: $(wildcard src/*cpp) $(wildcard ${STONNE_ROOT}/src/*cpp)  $(wildcard ${MRNA_ROOT}/src/*cpp)
	@mkdir -p $(@D)
	$(CC) $(PKG_CFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS)
