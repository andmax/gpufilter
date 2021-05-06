/*
PLR - Parallelized Linear Recurrences [float]
Copyright (c) 2018 Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted for academic, research, experimental, or personal use provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Texas State University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry Relations at Texas State University http://www.txstate.edu/ocir/.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Sepideh Maleki and Martin Burtscher


non-recursive coefficients: (0.040353)
recursive coefficients: (2.132330, -1.573120, 0.400436)

*/

#include <cstdio>
#include <cassert>
#include <cuda.h>

typedef float T;
static const int device = 0;
static const int order = 3;
static const int warp_size = 32;
static const int block_size = 1024;

static __device__ const T facA[300] = {4.004360e-01f, 8.538617e-01f, 1.190781e+00f, 1.356260e+00f, 1.360669e+00f, 1.244668e+00f, 1.056642e+00f, 8.399586e-01f, 6.272539e-01f, 4.392741e-01f, 2.862814e-01f, 1.705905e-01f, 8.930148e-02f, 3.669824e-02f, 6.081402e-03f, -9.003654e-03f, -1.407024e-02f, -1.340335e-02f, -1.005157e-02f, -5.982426e-03f, -2.311360e-03f, 4.574805e-04f, 2.215967e-03f, 3.079950e-03f, 3.264679e-03f, 3.003595e-03f, 2.502247e-03f, 1.917895e-03f, 1.355999e-03f, 8.763468e-04f, 5.035064e-04f, 2.380338e-04f, 6.641139e-05f, -3.122257e-05f, -7.573263e-05f, -8.577658e-05f, -7.627012e-05f, -5.802227e-05f, -3.808860e-05f, -2.048277e-05f, -6.992295e-06f, 2.059927e-06f, 7.190126e-06f, 9.291241e-06f, 9.325930e-06f, 8.148910e-06f, 6.425903e-06f, 4.617371e-06f, 3.000159e-06f, 1.706813e-06f, 7.688400e-07f, 1.557706e-07f, -1.938537e-07f, -3.505349e-07f, -3.801246e-07f, -3.367437e-07f, -2.604339e-07f, -1.778083e-07f, -1.042964e-07f, -4.696778e-08f, -7.280846e-09f, 1.659675e-08f, 2.803580e-08f, 3.075738e-08f, 2.812715e-08f, 2.281785e-08f, 1.672417e-08f, 1.102934e-08f, 6.346173e-09f, 2.878612e-09f, 5.714038e-10f, -7.687435e-10f, -1.385402e-09f, -1.515997e-09f, -1.361036e-09f, -1.072097e-09f, -7.520506e-10f, -4.620915e-10f, -2.315717e-10f, -6.801004e-11f, 3.423213e-11f, 8.725253e-11f, 1.049663e-10f, 1.002718e-10f, 8.362706e-11f, 6.261321e-11f, 4.210906e-11f, 2.477960e-11f, 1.166827e-11f, 2.761305e-12f, -2.544936e-12f, -5.098109e-12f, -5.761636e-12f, -5.284855e-12f, -4.246777e-12f, -3.048985e-12f, -1.936999e-12f, -1.034463e-12f, -3.796089e-13f, 4.223916e-14f, 2.730019e-13f, 3.636737e-13f, 3.629217e-13f, 3.110862e-13f, 2.380470e-13f, 1.635459e-13f, 9.882725e-14f, 4.877768e-14f, 1.403263e-14f, -7.236951e-15f, -1.797424e-14f, -2.132325e-14f, -2.009050e-14f, -1.649308e-14f, -1.210252e-14f, -7.905927e-15f, -4.423757e-15f, -1.842221e-15f, -1.349404e-16f, 8.388659e-16f, 1.263324e-15f, 1.320153e-15f, 1.163553e-15f, 9.101997e-16f, 6.390748e-16f, 3.967934e-16f, 2.052298e-16f, 6.932263e-17f, -1.614205e-17f, -6.129159e-17f, -7.754122e-17f, -7.538830e-17f, -6.331444e-17f, -4.746274e-17f, -3.179320e-17f, -1.848239e-17f, -8.401827e-18f, -1.571592e-18f, 2.464915e-18f, 4.363920e-18f, 4.798388e-18f, 4.353818e-18f, 3.482806e-18f, 2.498861e-18f, 1.592950e-18f, 8.603277e-19f, 3.292348e-19f, -1.348682e-20f, -2.021780e-19f, -2.780564e-19f, -2.802583e-19f, -2.411463e-19f, -1.846674e-19f, -1.266453e-19f, -7.610917e-20f, -3.700914e-20f, -9.900163e-21f, 6.632551e-21f, 1.489714e-20f, 1.736743e-20f, 1.625402e-20f, 1.330323e-20f, 9.751900e-21f, 6.375383e-21f, 3.580604e-21f, 1.510800e-21f, 1.417361e-22f, -6.406382e-22f, -9.840412e-22f, -1.033744e-21f, -9.128021e-22f, -7.142379e-22f, -5.009920e-22f, -3.102170e-22f, -1.593711e-22f, -5.243851e-23f, 1.467165e-23f, 4.995894e-23f, 6.245041e-23f, 6.044852e-23f, 5.065956e-23f, 3.793751e-23f, 2.540749e-23f, 1.478281e-23f, 6.744335e-24f, 1.300093e-24f, -1.917852e-24f, -3.434022e-24f, -3.784852e-24f, -3.436402e-24f, -2.748623e-24f, -1.970689e-24f, -1.254305e-24f, -6.751096e-25f, -2.555189e-25f, 1.490903e-26f, 1.634145e-25f, 2.226810e-25f, 2.237289e-25f, 1.921969e-25f, 1.470444e-25f, 1.007874e-25f, 6.055608e-26f, 2.945677e-26f, 7.908466e-27f, -5.226737e-27f, -1.179054e-26f, -1.375221e-26f, -1.286928e-26f, -1.052904e-26f, -7.713337e-27f, -5.037268e-27f, -2.823317e-27f, -1.184715e-27f, -1.018900e-28f, 5.158778e-28f, 7.859042e-28f, 8.234690e-28f, 7.261621e-28f, 5.677060e-28f, 3.979431e-28f, 2.462577e-28f, 1.264204e-28f, 4.152785e-29f, -1.171296e-29f, -3.968091e-29f, -4.955764e-29f, -4.794071e-29f, -4.015496e-29f, -3.005179e-29f, -2.010896e-29f, -1.168335e-29f, -5.312771e-30f, -1.001623e-30f, 1.543403e-30f, 2.739292e-30f, 3.012031e-30f, 2.731442e-30f, 2.182961e-30f, 1.564033e-30f, 9.947427e-31f, 5.348439e-31f, 2.019093e-31f, -1.250566e-32f, -1.301229e-31f, -1.769404e-31f, -1.776040e-31f, -1.524678e-31f, -1.165725e-31f, -7.983995e-32f, -4.791618e-32f, -2.325511e-32f, -6.180455e-33f, 4.216945e-33f, 9.402333e-33f, 1.094024e-32f, 1.022582e-32f, 8.359541e-33f, 6.119728e-33f, 3.993504e-33f, 2.235863e-33f, 9.358959e-34f, 7.750108e-35f, -4.116986e-34f, -6.250294e-34f, -6.540833e-34f, -5.763340e-34f, -4.502671e-34f, -3.153939e-34f, -1.949846e-34f, -9.992214e-35f, -3.262791e-35f, 9.537197e-36f, 3.165165e-35f, 3.942321e-35f, 3.809049e-35f, 3.187851e-35f, 2.384105e-35f, 1.594109e-35f, 9.252121e-36f, 4.198147e-36f, 7.805235e-37f, -1.234973e-36f, -2.180138e-36f, -2.393464e-36f, -2.168563e-36f, -1.731891e-36f, -1.239984e-36f, -7.879521e-37f, -4.230421e-37f, -1.590565e-37f, 0.000000e+00f, 1.038658e-37f, 1.407775e-37f, 1.411198e-37f, 1.210456e-37f, 9.248325e-38f, 6.329497e-38f, 3.794953e-38f, 1.838396e-38f};

static __device__ const T facB[307] = {-1.573120e+00f, -2.953975e+00f, -3.824143e+00f, -4.137311e+00f, -3.989155e+00f, -3.529032e+00f, -2.906369e+00f, -2.243149e+00f, -1.624217e+00f, -1.098439e+00f, -6.853842e-01f, -3.838832e-01f, -1.802289e-01f, -5.486554e-02f, 1.280957e-02f, 4.145417e-02f, 4.627284e-02f, 3.858599e-02f, 2.608508e-02f, 1.345091e-02f, 3.098036e-03f, -4.108451e-03f, -8.247929e-03f, -9.883654e-03f, -9.745402e-03f, -8.535006e-03f, -6.826533e-03f, -5.032243e-03f, -3.409170e-03f, -2.086742e-03f, -1.101681e-03f, -4.316070e-04f, -2.285797e-05f, 1.890759e-04f, 2.662995e-04f, 2.612462e-04f, 2.138548e-04f, 1.516733e-04f, 9.161064e-05f, 4.237898e-05f, 6.986898e-06f, -1.508465e-05f, -2.618662e-05f, -2.931073e-05f, -2.734590e-05f, -2.268726e-05f, -1.709540e-05f, -1.171354e-05f, -7.168809e-06f, -3.705080e-06f, -1.313579e-06f, 1.569026e-07f, 9.173381e-07f, 1.183236e-06f, 1.142797e-06f, 9.427829e-07f, 6.863776e-07f, 4.380899e-07f, 2.319221e-07f, 8.021678e-08f, -1.836572e-08f, -7.248244e-08f, -9.354329e-08f, -9.279589e-08f, -7.974120e-08f, -6.151359e-08f, -4.288362e-08f, -2.660501e-08f, -1.390185e-08f, -4.962594e-09f, 6.337801e-10f, 3.591385e-09f, 4.673803e-09f, 4.570199e-09f, 3.830839e-09f, 2.850701e-09f, 1.882337e-09f, 1.063276e-09f, 4.476357e-10f, 3.560213e-11f, -2.024952e-10f, -3.085435e-10f, -3.251110e-10f, -2.889544e-10f, -2.282593e-10f, -1.623504e-10f, -1.028131e-10f, -5.523802e-11f, -2.105931e-11f, 8.205728e-13f, 1.275926e-11f, 1.748318e-11f, 1.753665e-11f, 1.500006e-11f, 1.139870e-11f, 7.731212e-12f, 4.560532e-12f, 2.126885e-12f, 4.568121e-13f, -5.455694e-13f, -1.030273e-12f, -1.155712e-12f, -1.062082e-12f, -8.591936e-13f, -6.240907e-13f, -4.044484e-13f, -2.247000e-13f, -9.279703e-14f, -6.349494e-15f, 4.246365e-14f, 6.337576e-14f, 6.579505e-14f, 5.760304e-14f, 4.470313e-14f, 3.105203e-14f, 1.895612e-14f, 9.472874e-15f, 2.813390e-15f, -1.312176e-15f, -3.430513e-15f, -4.124192e-15f, -3.922972e-15f, -3.250923e-15f, -2.412210e-15f, -1.600436e-15f, -9.197477e-16f, -4.094635e-16f, -6.711000e-17f, 1.327344e-16f, 2.246417e-16f, 2.433299e-16f, 2.186228e-16f, 1.733436e-16f, 1.231438e-16f, 7.743747e-17f, 4.081522e-17f, 1.452430e-17f, -2.227892e-18f, -1.125518e-17f, -1.467896e-17f, -1.448677e-17f, -1.230578e-17f, -9.328549e-18f, -6.334096e-18f, -3.759134e-18f, -1.786908e-18f, -4.331085e-19f, 3.821979e-19f, 7.807613e-19f, 8.901652e-19f, 8.229406e-19f, 6.670891e-19f, 4.843241e-19f, 3.128625e-19f, 1.723527e-19f, 6.928337e-20f, 1.884938e-21f, -3.595551e-20f, -5.189069e-20f, -5.333094e-20f, -4.648677e-20f, -3.600806e-20f, -2.500742e-20f, -1.529405e-20f, -7.691217e-21f, -2.354704e-21f, 9.539129e-22f, 2.658448e-21f, 3.225160e-21f, 3.077030e-21f, 2.552217e-21f, 1.893101e-21f, 1.253928e-21f, 7.177111e-22f, 3.158841e-22f, 4.664129e-23f, -1.100716e-22f, -1.815900e-22f, -1.953771e-22f, -1.750222e-22f, -1.385686e-22f, -9.837913e-23f, -6.187691e-23f, -3.266768e-23f, -1.171301e-23f, 1.636430e-24f, 8.834065e-24f, 1.157253e-23f, 1.143469e-23f, 9.715038e-24f, 7.361581e-24f, 4.993261e-24f, 2.956881e-24f, 1.397890e-24f, 3.287158e-25f, -3.140768e-25f, -6.270591e-25f, -7.113869e-25f, -6.562400e-25f, -5.313204e-25f, -3.854710e-25f, -2.489028e-25f, -1.371105e-25f, -5.516745e-26f, -1.613523e-27f, 2.844046e-26f, 4.109167e-26f, 4.223464e-26f, 3.680464e-26f, 2.849406e-26f, 1.977289e-26f, 1.207565e-26f, 6.054199e-27f, 1.830872e-27f, -7.844301e-28f, -2.128526e-27f, -2.571570e-27f, -2.449123e-27f, -2.029288e-27f, -1.504097e-27f, -9.956340e-28f, -5.694955e-28f, -2.503949e-28f, -3.672758e-29f, 8.753944e-29f, 1.441727e-28f, 1.550067e-28f, 1.387783e-28f, 1.098091e-28f, 7.790441e-29f, 4.894691e-29f, 2.578950e-29f, 9.188076e-30f, -1.377857e-30f, -7.064948e-30f, -9.218029e-30f, -9.093611e-30f, -7.718574e-30f, -5.844435e-30f, -3.961432e-30f, -2.343876e-30f, -1.106432e-30f, -2.583804e-31f, 2.510263e-31f, 4.986788e-31f, 5.649886e-31f, 5.207803e-31f, 4.213696e-31f, 3.054908e-31f, 1.970814e-31f, 1.084005e-31f, 4.344237e-32f, 1.024996e-33f, -2.274697e-32f, -3.272060e-32f, -3.357696e-32f, -2.923243e-32f, -2.261512e-32f, -1.568220e-32f, -9.569041e-33f, -4.790280e-33f, -1.440925e-33f, 6.313694e-34f, 1.694835e-33f, 2.043729e-33f, 1.944549e-33f, 1.610063e-33f, 1.192558e-33f, 7.887731e-34f, 4.506149e-34f, 1.975681e-34f, 2.826219e-35f, -7.009156e-35f, -1.148048e-34f, -1.232220e-34f, -1.102155e-34f, -8.714478e-35f, -6.178172e-35f, -3.878409e-35f, -2.040632e-35f, -7.240612e-36f, 1.131673e-36f, 5.632026e-36f, 7.329679e-36f, 7.222605e-36f, 6.125777e-36f, 4.635221e-36f, 3.139429e-36f, 1.855523e-36f, 8.739973e-37f, 2.018311e-37f, -2.015140e-37f, -3.972188e-37f, -4.491755e-37f, -4.136110e-37f, -3.344088e-37f, -2.422763e-37f, -1.561725e-37f, -8.579103e-38f, -3.427281e-38f, 0.000000e+00f, 1.815737e-38f, 2.602928e-38f, 2.667561e-38f, 2.320490e-38f, 1.793961e-38f, 1.243096e-38f};

static __device__ const T facC[306] = {2.132330e+00f, 2.973711e+00f, 3.386958e+00f, 3.397969e+00f, 3.108282e+00f, 2.638729e+00f, 2.097609e+00f, 1.566426e+00f, 1.096988e+00f, 7.149222e-01f, 4.260101e-01f, 2.230090e-01f, 9.164453e-02f, 1.518613e-02f, -2.248514e-02f, -3.513759e-02f, -3.347203e-02f, -2.510162e-02f, -1.493979e-02f, -5.772089e-03f, 1.142484e-03f, 5.533914e-03f, 7.691513e-03f, 8.152826e-03f, 7.500819e-03f, 6.248807e-03f, 4.789514e-03f, 3.386298e-03f, 2.188473e-03f, 1.257386e-03f, 5.944275e-04f, 1.658399e-04f, -7.797778e-05f, -1.891303e-04f, -2.142115e-04f, -1.904701e-04f, -1.448993e-04f, -9.511872e-05f, -5.115166e-05f, -1.746192e-05f, 5.144146e-06f, 1.795575e-05f, 2.320285e-05f, 2.328948e-05f, 2.035011e-05f, 1.604727e-05f, 1.153085e-05f, 7.492218e-06f, 4.262369e-06f, 1.919989e-06f, 3.889861e-07f, -4.841206e-07f, -8.753940e-07f, -9.492850e-07f, -8.409482e-07f, -6.503792e-07f, -4.440385e-07f, -2.604581e-07f, -1.172918e-07f, -1.818209e-08f, 4.144710e-08f, 7.001364e-08f, 7.681015e-08f, 7.024165e-08f, 5.698277e-08f, 4.176507e-08f, 2.754347e-08f, 1.584824e-08f, 7.188746e-09f, 1.426986e-09f, -1.919747e-09f, -3.459722e-09f, -3.785860e-09f, -3.398880e-09f, -2.677319e-09f, -1.878075e-09f, -1.153967e-09f, -5.782951e-10f, -1.698363e-10f, 8.549059e-11f, 2.178968e-10f, 2.621324e-10f, 2.504084e-10f, 2.088414e-10f, 1.563635e-10f, 1.051586e-10f, 6.188185e-11f, 2.913903e-11f, 6.895734e-12f, -6.355489e-12f, -1.273150e-11f, -1.438851e-11f, -1.319784e-11f, -1.060546e-11f, -7.614215e-12f, -4.837256e-12f, -2.583358e-12f, -9.479934e-13f, 1.054855e-13f, 6.817679e-13f, 9.082020e-13f, 9.063237e-13f, 7.768749e-13f, 5.944743e-13f, 4.084228e-13f, 2.468013e-13f, 1.218128e-13f, 3.504411e-14f, -1.807236e-14f, -4.488659e-14f, -5.325012e-14f, -5.017165e-14f, -4.118789e-14f, -3.022341e-14f, -1.974333e-14f, -1.104735e-14f, -4.600510e-15f, -3.369395e-16f, 2.094932e-15f, 3.154922e-15f, 3.296833e-15f, 2.905751e-15f, 2.273050e-15f, 1.595968e-15f, 9.909176e-16f, 5.125250e-16f, 1.731233e-16f, -4.030834e-17f, -1.530609e-16f, -1.936417e-16f, -1.882657e-16f, -1.581142e-16f, -1.185281e-16f, -7.939688e-17f, -4.615600e-17f, -2.098193e-17f, -3.924826e-18f, 6.155537e-18f, 1.089794e-17f, 1.198296e-17f, 1.087276e-17f, 8.697606e-18f, 6.240416e-18f, 3.978091e-18f, 2.148516e-18f, 8.222165e-19f, -3.366491e-20f, -5.048869e-19f, -6.943814e-19f, -6.998834e-19f, -6.022118e-19f, -4.611691e-19f, -3.162718e-19f, -1.900688e-19f, -9.242456e-20f, -2.472529e-20f, 1.656206e-20f, 3.720150e-20f, 4.337087e-20f, 4.059063e-20f, 3.322184e-20f, 2.435326e-20f, 1.592120e-20f, 8.941867e-21f, 3.772978e-21f, 3.540264e-22f, -1.599801e-21f, -2.457393e-21f, -2.581529e-21f, -2.279515e-21f, -1.783653e-21f, -1.251122e-21f, -7.747060e-22f, -3.980020e-22f, -1.309605e-22f, 3.663374e-23f, 1.247575e-22f, 1.559535e-22f, 1.509554e-22f, 1.265105e-22f, 9.474053e-23f, 6.344987e-23f, 3.691719e-23f, 1.684289e-23f, 3.247049e-24f, -4.789137e-24f, -8.575518e-24f, -9.451713e-24f, -8.581595e-24f, -6.864059e-24f, -4.921364e-24f, -3.132364e-24f, -1.685954e-24f, -6.381163e-25f, 3.722132e-26f, 4.080850e-25f, 5.560936e-25f, 5.587130e-25f, 4.799704e-25f, 3.672125e-25f, 2.516961e-25f, 1.512271e-25f, 7.356310e-26f, 1.975059e-26f, -1.305203e-26f, -2.944397e-26f, -3.434300e-26f, -3.213821e-26f, -2.629404e-26f, -1.926248e-26f, -1.257958e-26f, -7.050697e-27f, -2.958621e-27f, -2.544781e-28f, 1.288281e-27f, 1.962626e-27f, 2.056444e-27f, 1.813445e-27f, 1.417736e-27f, 9.937881e-28f, 6.149845e-28f, 3.157142e-28f, 1.037110e-28f, -2.924826e-29f, -9.909349e-29f, -1.237594e-28f, -1.197219e-28f, -1.002788e-28f, -7.504839e-29f, -5.021822e-29f, -2.917697e-29f, -1.326770e-29f, -2.501428e-30f, 3.854305e-30f, 6.840831e-30f, 7.521963e-30f, 6.821262e-30f, 5.451546e-30f, 3.905896e-30f, 2.484202e-30f, 1.335691e-30f, 5.042475e-31f, -3.121662e-32f, -3.249470e-31f, -4.418680e-31f, -4.435279e-31f, -3.807570e-31f, -2.911168e-31f, -1.993852e-31f, -1.196621e-31f, -5.807598e-32f, -1.543526e-32f, 1.053038e-32f, 2.348006e-32f, 2.732085e-32f, 2.553685e-32f, 2.087628e-32f, 1.528284e-32f, 9.973037e-33f, 5.583677e-33f, 2.337258e-33f, 1.935753e-34f, -1.028116e-33f, -1.560877e-33f, -1.633441e-33f, -1.439283e-33f, -1.124459e-33f, -7.876404e-34f, -4.869416e-34f, -2.495410e-34f, -8.148575e-35f, 2.381446e-35f, 7.904196e-35f, 9.845071e-35f, 9.512305e-35f, 7.961020e-35f, 5.953846e-35f, 3.980992e-35f, 2.310555e-35f, 1.048421e-35f, 1.949313e-36f, -3.084043e-36f, -5.444446e-36f, -5.977210e-36f, -5.415579e-36f, -4.325085e-36f, -3.096643e-36f, -1.967779e-36f, -1.056483e-36f, -3.972256e-37f, 2.698968e-38f, 2.593803e-37f, 3.515631e-37f, 3.524196e-37f, 3.022893e-37f, 2.309607e-37f, 1.580685e-37f, 9.477289e-38f, 4.591126e-38f, 1.210517e-38f, 0.000000e+00f, -1.870049e-38f, -2.171775e-38f, -2.027947e-38f, -1.656624e-38f, -1.211922e-38f};

// shared memory size is 15760 bytes

static __device__ unsigned int counter = 0;

static __global__ __launch_bounds__(block_size, 2)
void Recurrence1(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 1;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
  }

  val0 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
    }
  }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val0;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
  } else {
    output[offs + (0 * block_size)] = val0;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence2(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 2;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val1;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence3(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 3;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;
  val2 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 8);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 5, 16);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 13, 32);
  if (cond) val2 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val2 += sfacC[tid] * spartc[31 * order + (1 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val2;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence4(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 4;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;
  val2 *= 4.035270e-02f;
  val3 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 8);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 5, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 5, 16);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 13, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 13, 32);
  if (cond) val3 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
  val3 += sfacC[tid] * spartc[31 * order + (2 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val2 += sfacC[tid] * spartc[31 * order + (1 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val3;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence5(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 5;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;
  val2 *= 4.035270e-02f;
  val3 *= 4.035270e-02f;
  val4 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 8);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 5, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 5, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 5, 16);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 13, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 13, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 13, 32);
  if (cond) val4 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
  val3 += sfacC[tid] * spartc[31 * order + (2 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val2 += sfacC[tid] * spartc[31 * order + (1 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
  val4 += sfacC[tid] * spartc[31 * order + (3 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val4;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence6(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 6;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;
  val2 *= 4.035270e-02f;
  val3 *= 4.035270e-02f;
  val4 *= 4.035270e-02f;
  val5 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 8);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 5, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 5, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 5, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 5, 16);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 13, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 13, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 13, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 13, 32);
  if (cond) val5 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
  val3 += sfacC[tid] * spartc[31 * order + (2 * delta + 2)];
  val5 += sfacC[tid] * spartc[31 * order + (4 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val2 += sfacC[tid] * spartc[31 * order + (1 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
  val4 += sfacC[tid] * spartc[31 * order + (3 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val5;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence7(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 7;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5, val6;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
    val6 = 0;
    if (offs + (6 * block_size) < items) val6 = input[offs + (6 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
    val6 = input[offs + (6 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;
  val2 *= 4.035270e-02f;
  val3 *= 4.035270e-02f;
  val4 *= 4.035270e-02f;
  val5 *= 4.035270e-02f;
  val6 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 2);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 4);
  if (cond) val6 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 4);
  if (cond) val6 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 8);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 2, 8);
  if (cond) val6 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 3, 8);
  if (cond) val6 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 5, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 5, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 5, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 5, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 5, 16);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 6, 16);
  if (cond) val6 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 7, 16);
  if (cond) val6 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 13, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 13, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 13, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 13, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 13, 32);
  if (cond) val6 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 14, 32);
  if (cond) val6 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 15, 32);
  if (cond) val6 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
    spartc[clwo + 6 * delta] = val6;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
      spartc[clane + (15 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
  val3 += sfacC[tid] * spartc[31 * order + (2 * delta + 2)];
  val5 += sfacC[tid] * spartc[31 * order + (4 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val6 += sfacB[tid] * spartc[31 * order + (5 * delta + 1)];
  val2 += sfacC[tid] * spartc[31 * order + (1 * delta + 2)];
  val6 += sfacC[tid] * spartc[31 * order + (5 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
  val4 += sfacC[tid] * spartc[31 * order + (3 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val6;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
    if (offs + (6 * block_size) < items) output[offs + (6 * block_size)] = val6;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
    output[offs + (6 * block_size)] = val6;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence8(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 8;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5, val6, val7;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
    val6 = 0;
    if (offs + (6 * block_size) < items) val6 = input[offs + (6 * block_size)];
    val7 = 0;
    if (offs + (7 * block_size) < items) val7 = input[offs + (7 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
    val6 = input[offs + (6 * block_size)];
    val7 = input[offs + (7 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;
  val2 *= 4.035270e-02f;
  val3 *= 4.035270e-02f;
  val4 *= 4.035270e-02f;
  val5 *= 4.035270e-02f;
  val6 *= 4.035270e-02f;
  val7 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 2);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 2);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 4);
  if (cond) val7 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 1, 4);
  if (cond) val7 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 1, 8);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 2, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 2, 8);
  if (cond) val7 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 3, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 3, 8);
  if (cond) val7 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 5, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 5, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 5, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 5, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 5, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 5, 16);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 6, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 6, 16);
  if (cond) val7 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 7, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 7, 16);
  if (cond) val7 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 13, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 13, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 13, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 13, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 13, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 13, 32);
  if (cond) val7 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 14, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 14, 32);
  if (cond) val7 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 15, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 15, 32);
  if (cond) val7 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
    spartc[clwo + 6 * delta] = val6;
    spartc[clwo + 7 * delta] = val7;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
      spartc[clane + (15 * order + 6 * delta)] = val6;
      spartc[clane + (15 * order + 7 * delta)] = val7;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val7 += sfacA[tid] * spartc[31 * order + (6 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
  val7 += sfacB[tid] * spartc[31 * order + (6 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
  val3 += sfacC[tid] * spartc[31 * order + (2 * delta + 2)];
  val5 += sfacC[tid] * spartc[31 * order + (4 * delta + 2)];
  val7 += sfacC[tid] * spartc[31 * order + (6 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val6 += sfacB[tid] * spartc[31 * order + (5 * delta + 1)];
  val2 += sfacC[tid] * spartc[31 * order + (1 * delta + 2)];
  val6 += sfacC[tid] * spartc[31 * order + (5 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
  val4 += sfacC[tid] * spartc[31 * order + (3 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val7;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
    if (offs + (6 * block_size) < items) output[offs + (6 * block_size)] = val6;
    if (offs + (7 * block_size) < items) output[offs + (7 * block_size)] = val7;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
    output[offs + (6 * block_size)] = val6;
    output[offs + (7 * block_size)] = val7;
  }
}

static __global__ __launch_bounds__(block_size, 2)
void Recurrence9(const int items, const T* const __restrict__ input, T* const __restrict__ output, volatile int* const __restrict__ status, volatile T* const __restrict__ partcarry, volatile T* const __restrict__ fullcarry)
{
  const int valsperthread = 9;
  const int chunk_size = valsperthread * block_size;
  __shared__ T spartc[chunk_size / warp_size * order];
  __shared__ T sfullc[order];
  __shared__ int cid;

  const int tid = threadIdx.x;
  const int warp = tid / warp_size;
  const int lane = tid % warp_size;

  __shared__ T sfacA[block_size];
  __shared__ T sfacB[block_size];
  __shared__ T sfacC[block_size];
  if (tid < 300) sfacA[tid] = facA[tid];
  else sfacA[tid] = 0;
  if (tid < 307) sfacB[tid] = facB[tid];
  else sfacB[tid] = 0;
  if (tid < 306) sfacC[tid] = facC[tid];
  else sfacC[tid] = 0;

  if (tid == 0) {
    cid = atomicInc(&counter, gridDim.x - 1);
  }

  __syncthreads();
  const int chunk_id = cid;
  const int offs = tid + chunk_id * chunk_size;

  T val0, val1, val2, val3, val4, val5, val6, val7, val8;
  if (chunk_id == gridDim.x - 1) {
    val0 = 0;
    if (offs + (0 * block_size) < items) val0 = input[offs + (0 * block_size)];
    val1 = 0;
    if (offs + (1 * block_size) < items) val1 = input[offs + (1 * block_size)];
    val2 = 0;
    if (offs + (2 * block_size) < items) val2 = input[offs + (2 * block_size)];
    val3 = 0;
    if (offs + (3 * block_size) < items) val3 = input[offs + (3 * block_size)];
    val4 = 0;
    if (offs + (4 * block_size) < items) val4 = input[offs + (4 * block_size)];
    val5 = 0;
    if (offs + (5 * block_size) < items) val5 = input[offs + (5 * block_size)];
    val6 = 0;
    if (offs + (6 * block_size) < items) val6 = input[offs + (6 * block_size)];
    val7 = 0;
    if (offs + (7 * block_size) < items) val7 = input[offs + (7 * block_size)];
    val8 = 0;
    if (offs + (8 * block_size) < items) val8 = input[offs + (8 * block_size)];
  } else {
    val0 = input[offs + (0 * block_size)];
    val1 = input[offs + (1 * block_size)];
    val2 = input[offs + (2 * block_size)];
    val3 = input[offs + (3 * block_size)];
    val4 = input[offs + (4 * block_size)];
    val5 = input[offs + (5 * block_size)];
    val6 = input[offs + (6 * block_size)];
    val7 = input[offs + (7 * block_size)];
    val8 = input[offs + (8 * block_size)];
  }

  val0 *= 4.035270e-02f;
  val1 *= 4.035270e-02f;
  val2 *= 4.035270e-02f;
  val3 *= 4.035270e-02f;
  val4 *= 4.035270e-02f;
  val5 *= 4.035270e-02f;
  val6 *= 4.035270e-02f;
  val7 *= 4.035270e-02f;
  val8 *= 4.035270e-02f;

  const T sfA = sfacA[lane];
  const T sfB = sfacB[lane];
  const T sfC = sfacC[lane];

  int cond;
  T help, spc;

  help = 2.132330e+00f;
  cond = ((lane & 1) != 0);
  spc = help * __shfl(val0, 0, 2);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 2);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 2);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 2);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 2);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 2);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 2);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 2);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 0, 2);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 0, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 0, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 0, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 0, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 0, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 0, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 0, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 0, 4);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 0, 4);
  if (cond) val8 += spc;

  help = __shfl(sfC, lane % 2);
  cond = ((lane & 2) != 0);
  spc = help * __shfl(val0, 1, 4);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 4);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 4);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 4);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 4);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 4);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 4);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 1, 4);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 1, 4);
  if (cond) val8 += spc;

  help = __shfl(sfA, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 1, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 1, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 1, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 1, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 1, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 1, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 1, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 1, 8);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 1, 8);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 2, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 2, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 2, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 2, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 2, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 2, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 2, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 2, 8);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 2, 8);
  if (cond) val8 += spc;

  help = __shfl(sfC, lane % 4);
  cond = ((lane & 4) != 0);
  spc = help * __shfl(val0, 3, 8);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 3, 8);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 3, 8);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 3, 8);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 3, 8);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 3, 8);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 3, 8);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 3, 8);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 3, 8);
  if (cond) val8 += spc;

  help = __shfl(sfA, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 5, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 5, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 5, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 5, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 5, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 5, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 5, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 5, 16);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 5, 16);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 6, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 6, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 6, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 6, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 6, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 6, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 6, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 6, 16);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 6, 16);
  if (cond) val8 += spc;

  help = __shfl(sfC, lane % 8);
  cond = ((lane & 8) != 0);
  spc = help * __shfl(val0, 7, 16);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 7, 16);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 7, 16);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 7, 16);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 7, 16);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 7, 16);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 7, 16);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 7, 16);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 7, 16);
  if (cond) val8 += spc;

  help = __shfl(sfA, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 13, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 13, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 13, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 13, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 13, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 13, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 13, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 13, 32);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 13, 32);
  if (cond) val8 += spc;

  help = __shfl(sfB, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 14, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 14, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 14, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 14, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 14, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 14, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 14, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 14, 32);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 14, 32);
  if (cond) val8 += spc;

  help = __shfl(sfC, lane % 16);
  cond = ((lane & 16) != 0);
  spc = help * __shfl(val0, 15, 32);
  if (cond) val0 += spc;
  spc = help * __shfl(val1, 15, 32);
  if (cond) val1 += spc;
  spc = help * __shfl(val2, 15, 32);
  if (cond) val2 += spc;
  spc = help * __shfl(val3, 15, 32);
  if (cond) val3 += spc;
  spc = help * __shfl(val4, 15, 32);
  if (cond) val4 += spc;
  spc = help * __shfl(val5, 15, 32);
  if (cond) val5 += spc;
  spc = help * __shfl(val6, 15, 32);
  if (cond) val6 += spc;
  spc = help * __shfl(val7, 15, 32);
  if (cond) val7 += spc;
  spc = help * __shfl(val8, 15, 32);
  if (cond) val8 += spc;

  const int delta = block_size / warp_size * order;
  const int clane = lane - (warp_size - order);
  const int clwo = clane + warp * order;

  if (((warp & 1) == 0) && (clane >= 0)) {
    spartc[clwo + 0 * delta] = val0;
    spartc[clwo + 1 * delta] = val1;
    spartc[clwo + 2 * delta] = val2;
    spartc[clwo + 3 * delta] = val3;
    spartc[clwo + 4 * delta] = val4;
    spartc[clwo + 5 * delta] = val5;
    spartc[clwo + 6 * delta] = val6;
    spartc[clwo + 7 * delta] = val7;
    spartc[clwo + 8 * delta] = val8;
  }

  __syncthreads();
  if ((warp & 1) != 0) {
    const int cwarp = ((warp & ~1) | 0) * order;
    const T helpA = sfacA[tid % (warp_size * 1)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 1)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 1)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    val8 += helpC * spartc[cwarp + (8 * delta + 2)];
    if (((warp & 3) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
      spartc[clwo + 8 * delta] = val8;
    }
  }

  __syncthreads();
  if ((warp & 2) != 0) {
    const int cwarp = ((warp & ~3) | 1) * order;
    const T helpA = sfacA[tid % (warp_size * 2)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 2)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 2)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    val8 += helpC * spartc[cwarp + (8 * delta + 2)];
    if (((warp & 7) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
      spartc[clwo + 8 * delta] = val8;
    }
  }

  __syncthreads();
  if ((warp & 4) != 0) {
    const int cwarp = ((warp & ~7) | 3) * order;
    const T helpA = sfacA[tid % (warp_size * 4)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 4)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 4)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    val8 += helpC * spartc[cwarp + (8 * delta + 2)];
    if (((warp & 15) != 0) && (clane >= 0)) {
      spartc[clwo + 0 * delta] = val0;
      spartc[clwo + 1 * delta] = val1;
      spartc[clwo + 2 * delta] = val2;
      spartc[clwo + 3 * delta] = val3;
      spartc[clwo + 4 * delta] = val4;
      spartc[clwo + 5 * delta] = val5;
      spartc[clwo + 6 * delta] = val6;
      spartc[clwo + 7 * delta] = val7;
      spartc[clwo + 8 * delta] = val8;
    }
  }

  __syncthreads();
  if ((warp & 8) != 0) {
    const int cwarp = ((warp & ~15) | 7) * order;
    const T helpA = sfacA[tid % (warp_size * 8)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 8)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 8)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    val8 += helpC * spartc[cwarp + (8 * delta + 2)];
    if ((warp == 15) && (clane >= 0)) {
      spartc[clane + (15 * order + 0 * delta)] = val0;
      spartc[clane + (15 * order + 1 * delta)] = val1;
      spartc[clane + (15 * order + 2 * delta)] = val2;
      spartc[clane + (15 * order + 3 * delta)] = val3;
      spartc[clane + (15 * order + 4 * delta)] = val4;
      spartc[clane + (15 * order + 5 * delta)] = val5;
      spartc[clane + (15 * order + 6 * delta)] = val6;
      spartc[clane + (15 * order + 7 * delta)] = val7;
      spartc[clane + (15 * order + 8 * delta)] = val8;
    }
  }

  __syncthreads();
  if ((warp & 16) != 0) {
   if ((warp & 15) < 10) {
    const int cwarp = 15 * order;
    const T helpA = sfacA[tid % (warp_size * 16)];
    val0 += helpA * spartc[cwarp + (0 * delta + 0)];
    val1 += helpA * spartc[cwarp + (1 * delta + 0)];
    val2 += helpA * spartc[cwarp + (2 * delta + 0)];
    val3 += helpA * spartc[cwarp + (3 * delta + 0)];
    val4 += helpA * spartc[cwarp + (4 * delta + 0)];
    val5 += helpA * spartc[cwarp + (5 * delta + 0)];
    val6 += helpA * spartc[cwarp + (6 * delta + 0)];
    val7 += helpA * spartc[cwarp + (7 * delta + 0)];
    val8 += helpA * spartc[cwarp + (8 * delta + 0)];
    const T helpB = sfacB[tid % (warp_size * 16)];
    val0 += helpB * spartc[cwarp + (0 * delta + 1)];
    val1 += helpB * spartc[cwarp + (1 * delta + 1)];
    val2 += helpB * spartc[cwarp + (2 * delta + 1)];
    val3 += helpB * spartc[cwarp + (3 * delta + 1)];
    val4 += helpB * spartc[cwarp + (4 * delta + 1)];
    val5 += helpB * spartc[cwarp + (5 * delta + 1)];
    val6 += helpB * spartc[cwarp + (6 * delta + 1)];
    val7 += helpB * spartc[cwarp + (7 * delta + 1)];
    val8 += helpB * spartc[cwarp + (8 * delta + 1)];
    const T helpC = sfacC[tid % (warp_size * 16)];
    val0 += helpC * spartc[cwarp + (0 * delta + 2)];
    val1 += helpC * spartc[cwarp + (1 * delta + 2)];
    val2 += helpC * spartc[cwarp + (2 * delta + 2)];
    val3 += helpC * spartc[cwarp + (3 * delta + 2)];
    val4 += helpC * spartc[cwarp + (4 * delta + 2)];
    val5 += helpC * spartc[cwarp + (5 * delta + 2)];
    val6 += helpC * spartc[cwarp + (6 * delta + 2)];
    val7 += helpC * spartc[cwarp + (7 * delta + 2)];
    val8 += helpC * spartc[cwarp + (8 * delta + 2)];
   }
    if ((warp == 31) && (clane >= 0)) {
      spartc[clane + (31 * order + 0 * delta)] = val0;
      spartc[clane + (31 * order + 2 * delta)] = val2;
      spartc[clane + (31 * order + 4 * delta)] = val4;
      spartc[clane + (31 * order + 6 * delta)] = val6;
      spartc[clane + (31 * order + 8 * delta)] = val8;
    }
  }

  __syncthreads();
 if (warp < 10) {
  val1 += sfacA[tid] * spartc[31 * order + (0 * delta + 0)];
  val3 += sfacA[tid] * spartc[31 * order + (2 * delta + 0)];
  val5 += sfacA[tid] * spartc[31 * order + (4 * delta + 0)];
  val7 += sfacA[tid] * spartc[31 * order + (6 * delta + 0)];
  val1 += sfacB[tid] * spartc[31 * order + (0 * delta + 1)];
  val3 += sfacB[tid] * spartc[31 * order + (2 * delta + 1)];
  val5 += sfacB[tid] * spartc[31 * order + (4 * delta + 1)];
  val7 += sfacB[tid] * spartc[31 * order + (6 * delta + 1)];
  val1 += sfacC[tid] * spartc[31 * order + (0 * delta + 2)];
  val3 += sfacC[tid] * spartc[31 * order + (2 * delta + 2)];
  val5 += sfacC[tid] * spartc[31 * order + (4 * delta + 2)];
  val7 += sfacC[tid] * spartc[31 * order + (6 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 1 * delta)] = val1;
    spartc[clane + (31 * order + 5 * delta)] = val5;
  }

  __syncthreads();
 if (warp < 10) {
  val2 += sfacA[tid] * spartc[31 * order + (1 * delta + 0)];
  val6 += sfacA[tid] * spartc[31 * order + (5 * delta + 0)];
  val2 += sfacB[tid] * spartc[31 * order + (1 * delta + 1)];
  val6 += sfacB[tid] * spartc[31 * order + (5 * delta + 1)];
  val2 += sfacC[tid] * spartc[31 * order + (1 * delta + 2)];
  val6 += sfacC[tid] * spartc[31 * order + (5 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 3 * delta)] = val3;
  }

  __syncthreads();
 if (warp < 10) {
  val4 += sfacA[tid] * spartc[31 * order + (3 * delta + 0)];
  val4 += sfacB[tid] * spartc[31 * order + (3 * delta + 1)];
  val4 += sfacC[tid] * spartc[31 * order + (3 * delta + 2)];
 }
  if ((warp == 31) && (clane >= 0)) {
    spartc[clane + (31 * order + 7 * delta)] = val7;
  }

  __syncthreads();
 if (warp < 10) {
  val8 += sfacA[tid] * spartc[31 * order + (7 * delta + 0)];
  val8 += sfacB[tid] * spartc[31 * order + (7 * delta + 1)];
  val8 += sfacC[tid] * spartc[31 * order + (7 * delta + 2)];
 }

  const int idx = tid - (block_size - order);
  if (idx >= 0) {
    fullcarry[chunk_id * order + idx] = val8;
    __threadfence();
    if (idx == 0) {
      status[chunk_id] = 2;
    }
  }

  if (chunk_id > 0) {
    __syncthreads();
    if (warp == 0) {
      const int cidm1 = chunk_id - 1;
      int flag = 1;
      do {
        if ((cidm1 - lane) >= 0) {
          flag = status[cidm1 - lane];
        }
      } while ((__any(flag == 0)) || (__all(flag != 2)));
      int mask = __ballot(flag == 2);
      const int pos = __ffs(mask) - 1;
      T fc;
      if (lane < order) {
        fc = fullcarry[(cidm1 - pos) * order + lane];
      }
      T X0 = __shfl(fc, 0);
      T X1 = __shfl(fc, 1);
      T X2 = __shfl(fc, 2);
      if (lane == 0) {
        sfullc[0] = X0;
        sfullc[1] = X1;
        sfullc[2] = X2;
      }
    }

    __syncthreads();
    T X0 = sfullc[0];
    val0 += sfacA[tid] * X0;
    T X1 = sfullc[1];
    val0 += sfacB[tid] * X1;
    T X2 = sfullc[2];
    val0 += sfacC[tid] * X2;
  }

  if (chunk_id == gridDim.x - 1) {
    if (offs + (0 * block_size) < items) output[offs + (0 * block_size)] = val0;
    if (offs + (1 * block_size) < items) output[offs + (1 * block_size)] = val1;
    if (offs + (2 * block_size) < items) output[offs + (2 * block_size)] = val2;
    if (offs + (3 * block_size) < items) output[offs + (3 * block_size)] = val3;
    if (offs + (4 * block_size) < items) output[offs + (4 * block_size)] = val4;
    if (offs + (5 * block_size) < items) output[offs + (5 * block_size)] = val5;
    if (offs + (6 * block_size) < items) output[offs + (6 * block_size)] = val6;
    if (offs + (7 * block_size) < items) output[offs + (7 * block_size)] = val7;
    if (offs + (8 * block_size) < items) output[offs + (8 * block_size)] = val8;
  } else {
    output[offs + (0 * block_size)] = val0;
    output[offs + (1 * block_size)] = val1;
    output[offs + (2 * block_size)] = val2;
    output[offs + (3 * block_size)] = val3;
    output[offs + (4 * block_size)] = val4;
    output[offs + (5 * block_size)] = val5;
    output[offs + (6 * block_size)] = val6;
    output[offs + (7 * block_size)] = val7;
    output[offs + (8 * block_size)] = val8;
  }
}

struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001 * ms;}
};

int main(int argc, char *argv[])
{
  printf("Parallel Linear Recurrence Computation\n");
  printf("Copyright (c) 2018 Texas State University\n");

  if (argc != 2) {
    fprintf(stderr, "USAGE: %s problem_size\n", argv[0]);
    return -1;
  }

  const int n = atoi(argv[1]);
  if (n < 1) {fprintf(stderr, "ERROR: problem_size must be at least 1\n");  return -1;};

  int *d_status;
  T *h_in, *h_out, *h_sol, *d_in, *d_out, *d_partcarry, *d_fullcarry;

  const size_t size = n * sizeof(T);
  h_in = (T *)malloc(size);  assert(h_in != NULL);
  h_out = (T *)malloc(size);  assert(h_out != NULL);
  h_sol = (T *)malloc(size);  assert(h_sol != NULL);

  for (int i = 0; i < n; i++) {
    h_in[i] = (i & 32) / 16 - 1;
    h_sol[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    if ((i - 0) >= 0) {
      h_sol[i] += 4.035270e-02f * h_in[i - 0];
    }
  }
  for (int i = 1; i < n; i++) {
    if ((i - 1) >= 0) {
      h_sol[i] += 2.132330e+00f * h_sol[i - 1];
    }
    if ((i - 2) >= 0) {
      h_sol[i] += -1.573120e+00f * h_sol[i - 2];
    }
    if ((i - 3) >= 0) {
      h_sol[i] += 4.004360e-01f * h_sol[i - 3];
    }
  }

  cudaSetDevice(device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  const int SMs = deviceProp.multiProcessorCount;
  int valsperthread = 1;
  while ((valsperthread < 9) && (block_size * 2 * SMs * valsperthread < n)) {
    valsperthread++;
  }
  const int chunk_size = valsperthread * block_size;
  const int iterations = 5;

  assert(cudaSuccess == cudaMalloc(&d_in, size));
  assert(cudaSuccess == cudaMalloc(&d_out, size));
  assert(cudaSuccess == cudaMalloc(&d_status, (n + chunk_size - 1) / chunk_size * sizeof(int)));
  assert(cudaSuccess == cudaMalloc(&d_partcarry, (n + chunk_size - 1) / chunk_size * order * sizeof(T)));
  assert(cudaSuccess == cudaMalloc(&d_fullcarry, (n + chunk_size - 1) / chunk_size * order * sizeof(T)));
  assert(cudaSuccess == cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice));

  cudaMemset(d_status, 0, (n + chunk_size - 1) / chunk_size * sizeof(int));
  switch (valsperthread) {
    case 1:  Recurrence1<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 2:  Recurrence2<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 3:  Recurrence3<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 4:  Recurrence4<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 5:  Recurrence5<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 6:  Recurrence6<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 7:  Recurrence7<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 8:  Recurrence8<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    case 9:  Recurrence9<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
  }
  GPUTimer timer;
  timer.start();
  for (long i = 0; i < iterations; i++) {
    cudaMemset(d_status, 0, (n + chunk_size - 1) / chunk_size * sizeof(int));
    switch (valsperthread) {
      case 1:  Recurrence1<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 2:  Recurrence2<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 3:  Recurrence3<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 4:  Recurrence4<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 5:  Recurrence5<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 6:  Recurrence6<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 7:  Recurrence7<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 8:  Recurrence8<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
      case 9:  Recurrence9<<<(n + chunk_size - 1) / chunk_size, block_size>>>(n, d_in, d_out, d_status, d_partcarry, d_fullcarry); break;
    }
  }
  double runtime = timer.stop() / iterations;
  double throughput = 0.000000001 * n / runtime;
  assert(cudaSuccess == cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++) {
    T s = h_sol[i];
    T o = h_out[i];
    if (fabsf(o - s) > 0.001) {
      printf("result not correct at index %d: %e != %e\n", i, h_sol[i], h_out[i]);
      return -1;
    }
  }
  printf("size = %d\tthroughput = %7.4f gigaitems/s\truntime = %7.4f s\tPassed!\n", n, throughput, runtime);

  printf("first elements of result are:\n");
  for (int i = 0; (i < 8) && (i < n); i++) {
    printf(" %f", h_out[i]);
  }
  printf("\n");

  free(h_in);  free(h_out);  free(h_sol);
  cudaFree(d_in);  cudaFree(d_out);  cudaFree(d_status);  cudaFree(d_partcarry);  cudaFree(d_fullcarry);

  return 0;
}
