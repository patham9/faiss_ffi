g++ -std=c++17 -fPIC -shared faiss.cpp \
    -o faisslib.so \
    /usr/local/lib/libfaiss.a \
    -I/usr/local/include \
    -lopenblas -lblas -llapack -lgfortran -lm -lpthread \
    -fopenmp -lgomp \
    $(pkg-config --cflags --libs swipl)
echo "Successfully built faiss_ffi"
