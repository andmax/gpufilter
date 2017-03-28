
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>

#define RSPROP 2 // radius / sigma proportion

// auxiliary reflect function
inline
int reflect( const int& i, const int& n ) {

	if( i < 0 ) return (-i-1);
	else if( i >= n ) return (2*n-i-1);
	else return i;

}

// auxiliary clamp function
inline
int clamp( const int& i, const int& n ) {

	if( i < 0 ) return 0;
	else if( i >= n ) return n-1;
	else return i;

}

///------------- GAUSS-CPU - DIRECT APPROACH -------------
template< class T >
void gauss_cpu( T *out, const T* in, const int& w, const int& h, const std::vector< T >& ker, const int& kerR ) {

	T *buf = new T[ w * h ];

	for (int y = 0; y < h; ++y) {

		for (int x = 0; x < w; ++x) {

			T s = (T)0;

			for (int k = -kerR; k <= kerR; ++k) {

				int d = x + k;

				s += in[ y * w + clamp(d, w) ] * ker[ k + kerR ];

			}

			buf[ y * w + x ] = s;

		}

	}

	for (int x = 0; x < w; ++x) {

		for (int y = 0; y < h; ++y) {

			T s = (T)0;

			for (int k = -kerR; k <= kerR; ++k) {

				int d = y + k;

				s += buf[ clamp(d, h) * w + x ] * ker[ k + kerR ];

			}

			out[ y * w + x ] = s;

		}

	}

	delete [] buf;

}

///----------------- COMPUTE GAUSS KERNEL ----------------
template< class T >
bool compute_gauss( std::vector< T >& kernel, const int& radius ) {

	if( radius < 1 ) { std::cerr << "[error] Wrong kernel radius " << radius << "\n"; return false; }

	kernel.clear();
	kernel.resize( radius * 2 + 1 );

	T sigma = (T)radius / (T)RSPROP;
	T cutoff = (T)2 * sigma * sigma;
	T s = (T)0;

	for (int i = -radius; i <= radius; ++i) {

		T w = std::exp( -i * i / cutoff );

		s += w;

		kernel[ i + radius ] = w;

	}

	for (int i = -radius; i <= radius; ++i)
		kernel[ i + radius ] /= s;

	return true;

}
