LUAINC = -I/usr/local/include
LUALIB = -L./ -llua53

CC= gcc
CXX = g++
CFLAGS = -g -Wall

math3d : math3d.dll

math3d.dll : libmath3d.cpp
	$(CC) $(CFLAGS) libmath3d.cpp -shared -DLUA_BUILD_AS_DLL -o math3d.dll $(LUAINC) $(LUALIB) -I.
