/* CUDA_ANFISblimpCtrl.cu */

//-------------------------------| 
//          LIBRARIES            |
//-------------------------------| 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

//-------------------------------| 
//       GLOBAL VARIABLES        |
//-------------------------------| 
#define N 9
const double pi = 3.14159265359;
const double tol = 1e-6;
const int mInputs = 3;
const int mStates = 6;
const int nData = 6000;

const int nInputs = 4;
const int nFuzzy = 5;
const int nRules = 625;

const int Np = 4;
const int Nc = 3;

double tt[nData];
double UT[mInputs][nData];
double U[mInputs][nData];
double X[mStates][nData];
double Y[mStates][nData];
double Vearth[3][nData];
double Xearth[3][nData];
double lambda[3][3];
double dX[mStates][nData];
double Vtot[nData];
double Wtot[nData];
double G[mStates][nData];
double A[mStates][nData];
double Fd[mStates][nData];
double P[mStates][nData];

double OUTPUT[nData];
double INPUT1[nData];
double INPUT2[nData];
double INPUT3[nData];
double INPUT4[nData];

double O5[nData];
double En[nData];
double muIn[nFuzzy][nInputs];
double w[nRules];
double wn[nRules];
double fi[nRules];
double fi_wn[nRules];
double sumW;

double aIne[nFuzzy][nInputs];
double cIne[nFuzzy][nInputs];
double aOute[nRules];
double cOute[nRules];

double XX[nInputs];
double ukk[mInputs][nData];
double xkk[mStates][nData];
double ykk[mStates][nData];
double yee[nData];

double ref[nData];
double wref[nData];

double Q[Np][Np];
double R[Nc * mInputs][Nc * mInputs];

double gi[Np];
double Gij[Np][mInputs * Nc];
double Wref[Np];
double Yfree[Np];
double Ystep[Np][mInputs];

double Gijt[mInputs * Nc][Np];
double Gijt_Q[mInputs * Nc][Np];
double Gijt_Q_Gij[mInputs * Nc][mInputs * Nc];
double Gijt_Q_Gij_R[mInputs * Nc][mInputs * Nc];
double Gijt_Q_Gij_R_1[mInputs * Nc][mInputs * Nc];
double Gijt_Q_Gij_R_1_Gijt[mInputs * Nc][Np];
double Gijt_Q_Gij_R_1_Gijt_Q[mInputs * Nc][Np];
double Wref_Yfree[Np];

// Device
double* host_In = NULL;
double* host_Out = NULL;
double* host_aIn = NULL;
double* host_cIn = NULL;
double* host_aOut = NULL;
double* host_cOut = NULL;


double* dev_aIn = NULL;
double* dev_cIn = NULL;
double* dev_aOut = NULL;
double* dev_cOut = NULL;

double* dev_O0 = NULL;
double* dev_O1 = NULL;
double* dev_O2 = NULL;
double* dev_O3 = NULL;
double* dev_O4 = NULL;
double* dev_O5 = NULL;

const int acIn_size = nInputs * nFuzzy * sizeof(double);
const int acOut_size = nRules * sizeof(double);
const int O0_size = nInputs * sizeof(double);
const int O1_size = nInputs * nFuzzy * sizeof(double);
const int O2_size = nRules * sizeof(double);
const int O3_size = nRules * sizeof(double);
const int O4_size = nRules * sizeof(double);
const int O5_size = sizeof(double);
const int Input_size = nInputs * sizeof(double);
const int Output_size = sizeof(double);


//-------------------------------| 
//    FUNCTION DECLARATION       |
//-------------------------------| 

cudaError_t ANFIS_setup();
cudaError_t ANFIS_wrapper(double In[4]);

double h(double t);
double r(double t);
void fill(double* arr, int n, int m, double val);
double sign(double x);
double map(double input, double minIn, double maxIn, double minOut, double maxOut);
double reference(double t);
double gauss(double x, double a, double c);
double sigmoid(double x, double a, double c);
double invSigmoid(double x, double a, double c);
double dGauss_da(double x, double a, double c);
double dGauss_dc(double x, double a, double c);
double dinvSigmoid_da(double x, double a, double c);
double dinvSigmoid_dc(double x, double a, double c);
void matrixMultiplication(int a, double* A, int c, double* B, int b, double* C);
void matrixTranspose(int a, double* A, int b, double* B);
void matrixSum(int n, int m, double* A, double* B, double* C);
void matrixSubstract(int n, int m, double* A, double* B, double* C);

void set_ANFIS();

void getCfactor(double M[N][N], double t[N][N], int p, int q, int n);
double DET(double M[N][N], int n);
void ADJ(double M[N][N], double adj[N][N]);
bool INV(double M[N][N], double inv[N][N]);

void mINVERSE(double M[81], double inverse[81]);


//-------------------------------| 
//            KERNELS            |
//-------------------------------| 

__global__ void ANFIS(double* aIn, double* cIn, double* aOut, double* cOut, double* O0, double* O1, double* O2, double* O3, double* O4, double* O5) { // One 1024 thread block for 625 rules
    __shared__ double O0_shared[nInputs * nFuzzy];
    __shared__ double O1_shared[nInputs * nFuzzy];
    __shared__ double O2_shared[nRules];
    __shared__ double O3_shared[nRules];
    __shared__ double O4_shared[nRules];
    __shared__ double O5_shared;
    __shared__ double aIn_shared[nInputs * nFuzzy];
    __shared__ double cIn_shared[nInputs * nFuzzy];
    __shared__ double aOut_shared[nRules];
    __shared__ double cOut_shared[nRules];
    __shared__ double fi_shared[nRules];

    int index = threadIdx.x;
    const double dev_tol = 1e-6;
    __syncthreads();

    // Layer 1: Fuzzyfication
    if (index < nInputs * nFuzzy) {
        O0_shared[index] = O0[index % nInputs];
        aIn_shared[index] = aIn[index];
        cIn_shared[index] = cIn[index];

        O1_shared[index] = exp(-(O0_shared[index] - cIn_shared[index]) * (O0_shared[index] - cIn_shared[index]) / (aIn_shared[index] * aIn_shared[index]));

        if (O1_shared[index] < dev_tol) {
            O1_shared[index] = dev_tol;
        }
        if (O1_shared[index] > 1.0 - dev_tol) {
            O1_shared[index] = 1.0 - dev_tol;
        }
        O1[index] = O1_shared[index];

    }
    __syncthreads();

    // Layer 2: Permutation
    if (index < nRules) {
        O2_shared[index] = 1.0;
        O2_shared[index] = O2_shared[index] * O1_shared[index / (nFuzzy * nFuzzy * nFuzzy) * nInputs];
        O2_shared[index] = O2_shared[index] * O1_shared[(index / (nFuzzy * nFuzzy)) % nFuzzy * nInputs + 1];
        O2_shared[index] = O2_shared[index] * O1_shared[((index / nFuzzy) % nFuzzy) % nFuzzy * nInputs + 2];
        O2_shared[index] = O2_shared[index] * O1_shared[((index % nFuzzy) % nFuzzy) % nFuzzy * nInputs + 3];

        O2[index] = O2_shared[index];
        // O2[index] = O1_shared[index / (nFuzzy * nFuzzy * nFuzzy)*nInputs];
    }
    __syncthreads();

    // Layer 3: Normalization
    double sumW = 0.0;
    if (index < nRules) {
        for (int i = 0; i < nRules; i++) {
            sumW = sumW + O2_shared[i];
        }
        if (sqrt(sumW * sumW) < dev_tol) {
            sumW = dev_tol;
        }
        O3_shared[index] = O2_shared[index] / sumW;

        O3[index] = O3_shared[index];
    }
    __syncthreads();

    // Layer 4: Defuzzyfication
    if (index < nRules) {
        aOut_shared[index] = aOut[index];
        cOut_shared[index] = cOut[index];
        fi_shared[index] = cOut_shared[index] - aOut_shared[index] * log((1.0 / O2_shared[index]) - 1.0);
        O4_shared[index] = O3_shared[index] * fi_shared[index];
        O4[index] = O4_shared[index];
    }
    __syncthreads();

    // Layer 5: Output
    double O5_reg = 0.0;
    if (index < nRules) {
        for (int i = 0; i < nRules; i++) {
            O5_reg = O5_reg + O4_shared[i];
        }
        O5_shared = O5_reg;
        O5[0] = O5_shared;
    }
    __syncthreads();

}

/*
__global__ void ANFIS(double* aIn, double* cIn, double* aOut, double* cOut, double* O0, double* O1, double* O2, double* O3, double* O4, double* O5) { // One 1024 thread block for 625 rules
    __shared__ double fi_shared[nRules];

    int index = threadIdx.x;
    const double dev_tol = 1e-6;

    // Layer 1: Fuzzyfication
    if (index < nInputs * nFuzzy) {
        O1 [index] = exp(-(O0[index / nFuzzy] - cIn[index]) * (O0[index / nFuzzy] - cIn[index]) / (aIn[index] * aIn[index]));
        if (O1[index] < dev_tol) {
            O1[index] = dev_tol;
        }
        if (O1[index] > 1.0 - dev_tol) {
            O1[index] = 1.0 - dev_tol;
        }
    }
    __syncthreads();

    // Layer 2: Permutation
    if (index < nRules) {
        O2[index] = 1.0;
        O2[index] = O2[index] * O1[index / (nFuzzy * nFuzzy * nFuzzy) * nInputs];
        O2[index] = O2[index] * O1[(index / (nFuzzy * nFuzzy)) % nFuzzy * nInputs + 1];
        O2[index] = O2[index] * O1[((index / nFuzzy) % nFuzzy) % nFuzzy * nInputs + 2];
        O2[index] = O2[index] * O1[((index % nFuzzy) % nFuzzy) % nFuzzy * nInputs + 3];
    }
    __syncthreads();

    // Layer 3: Normalization
    double sumW = 0.0;
    if (index < nRules) {
        for (int i = 0; i < nRules; i++) {
            sumW = sumW + O2[i];
        }
        if (sqrt(sumW * sumW) < dev_tol) {
            sumW = dev_tol;
        }
        O3[index] = O2[index] / sumW;
    }
    __syncthreads();

    // Layer 4: Defuzzyfication
    if (index < nRules) {
        fi_shared[index] = cOut[index] - aOut[index] * log((1.0 / O2[index]) - 1.0);
        O4[index] = O3[index] * fi_shared[index];
    }
    __syncthreads();

    // Layer 5: Output
    double O5_reg = 0.0;
    if (index < nRules) {
        for (int i = 0; i < nRules; i++) {
            O5_reg = O5_reg + O4[i];
        }
        O5[0] = O5_reg;
    }
    __syncthreads();
}*/


//-------------------------------| 
//            MAIN()             |
//-------------------------------| 

int main()
{
    // Step 1: Loading ANFIS and setting up device
    printf("\n\nStep 1: Loading ANFIS and setting up device...");
    // a) Training parameters
    set_ANFIS();

    // b) Malloc for host
    host_In = (double*)malloc(Input_size);
    host_Out = (double*)malloc(Output_size);
    host_aIn = (double*)malloc(acIn_size);
    host_cIn = (double*)malloc(acIn_size);
    host_aOut = (double*)malloc(acOut_size);
    host_cOut = (double*)malloc(acOut_size);

    if (host_In == NULL) {
        printf("\n\t\tInsufficient memory available for host_In parameters.");
        goto Error;
    }

    if (host_Out == NULL) {
        printf("\n\t\tInsufficient memory available for host_Out parameters.");
        goto Error;
    }

    if (host_aIn == NULL) {
        printf("\n\t\tInsufficient memory available for host_aIn parameters.");
        goto Error;
    }

    if (host_cIn == NULL) {
        printf("\n\t\tInsufficient memory available for host_cIn parameters.");
        goto Error;
    }

    if (host_aOut == NULL) {
        printf("\n\t\tInsufficient memory available for host_aOut parameters.");
        goto Error;
    }

    if (host_cOut == NULL) {
        printf("\n\t\tInsufficient memory available for host_cOut parameters.");
        goto Error;
    }

    for (int i = 0; i < nInputs; i++) {
        for (int j = 0; j < nFuzzy; j++) {
            *(host_aIn + j * nInputs + i) = aIne[j][i];
            *(host_cIn + j * nInputs + i) = cIne[j][i];
        }
    }

    for (int i = 0; i < nRules; i++) {
        *(host_aOut + i) = aOute[i];
        *(host_cOut + i) = cOute[i];

    }

    // c) Setting up CUDA device
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = ANFIS_setup();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tANFIS_setup() failed!");
        goto Error;
    }


    // Step 2: Loading blimp parameters
    printf("\n\nStep 2: Loading blimp parameters...");
    // a) Physical constants
    printf("\n\ta) Physical constants");
    const double rhoAir = 1.205;            // Density of air at NTP (20°C, 1atm)
    const double rhoHe = 0.1664;           // Density of Helium at NTP (20°C, 1atm)
    const double g_acc = 9.80665;           // Acceleration of gravity
    const double deg2rad = pi / 180;          // Degrees to radians conversion 
    const double rad2deg = pow(deg2rad, -1.0); // Radians to degrees conversion

    // b) Vehicle geometry and parameters
    printf("\n\tb) Vehicle geometry and parameters");
    const double blimp_a = 0.9;                                     // Blimp's x length 
    const double blimp_b = 0.45;                                    // Blimp's y length 
    const double blimp_c = 0.45;                                    // Blimp's z length 
    const double blimp_volume = 4.0 * pi * blimp_a * blimp_b * blimp_c / 3.0;    // Blimp's volume 
    const double blimp_area = pow(blimp_volume, 0.6666666666);      // Blimp's area 
    const double blimp_mHe = blimp_volume * rhoHe;                    // Blimp's mass of helium 
    const double blimp_mAir = blimp_volume * rhoAir;                  // Blimp's mass of air
    const double blimp_mass = blimp_mAir - blimp_mHe;               // Blimp's mass (chosen for 0 buoyancy)
    const double blimp_mTotal = blimp_mass + blimp_mHe;               // Blimp's total mass
    const double blimp_dx = blimp_a / 8.0;                              // Blimp's x axis distace from CV to propellers
    const double blimp_dy = blimp_b / 2.0;                              // Blimp's y axis distace from CV to propellers
    const double blimp_dz = blimp_c;                                // Blimp's z axis distace from CV to propellers
    const double blimp_ax = 0.0;                                      // Blimp's x axis distance from center of gravity CG and center of volume CV
    //const double blimp_ay = 0.0;                                      // Blimp's y axis distance from center of gravity CG and center of volume CV
    const double blimp_az = -(0.2 * blimp_mass) * blimp_b / blimp_mTotal;   // Blimp's z axis distance from center of gravity CG and center of volume CV

    // c) Masses and inertias
    printf("\n\tc) Masses and inertias");
    const double blimp_Ix = blimp_mTotal * (blimp_b * blimp_b + blimp_c * blimp_c) / 5.0;
    const double blimp_Iy = blimp_mTotal * (blimp_c * blimp_c + blimp_a * blimp_a) / 5.0;
    const double blimp_Iz = blimp_mTotal * (blimp_a * blimp_a + blimp_b * blimp_b) / 5.0;

    // c.1) Tuckerman fo a prolate ellipsoid
    const double tuckerman_e = sqrt(1.0 - blimp_c * blimp_c / (blimp_a * blimp_a));
    const double tuckerman_alpha = (1.0 - tuckerman_e * tuckerman_e) * (log((1.0 + tuckerman_e) / (1.0 - tuckerman_e)) - 2.0 * tuckerman_e) / (tuckerman_e * tuckerman_e * tuckerman_e);
    const double tuckerman_beta = (1.0 - tuckerman_e * tuckerman_e) * ((tuckerman_e / (1.0 - tuckerman_e * tuckerman_e)) - 0.5 * log((1.0 + tuckerman_e) / (1.0 - tuckerman_e))) / (tuckerman_e * tuckerman_e * tuckerman_e);
    const double tuckerman_gamma = tuckerman_beta;

    const double tuckerman_K1 = blimp_volume * (tuckerman_alpha / (2.0 - tuckerman_alpha));
    const double tuckerman_K2 = blimp_volume * (tuckerman_beta / (2.0 - tuckerman_beta));
    const double tuckerman_K3 = blimp_volume * (tuckerman_gamma / (2.0 - tuckerman_gamma));
    const double tuckerman_K1_ = blimp_volume * blimp_Ix * (pow((blimp_b * blimp_b - blimp_c * blimp_c) / (blimp_b * blimp_b + blimp_c * blimp_c), 2.0) * ((tuckerman_gamma - tuckerman_beta) / (2.0 * ((blimp_b * blimp_b - blimp_c * blimp_c) / (blimp_b * blimp_b + blimp_c * blimp_c)) - (tuckerman_gamma - tuckerman_beta))));
    const double tuckerman_K2_ = blimp_volume * blimp_Iy * (pow((blimp_c * blimp_c - blimp_a * blimp_a) / (blimp_c * blimp_c + blimp_a * blimp_a), 2.0) * ((tuckerman_alpha - tuckerman_gamma) / (2.0 * ((blimp_c * blimp_c - blimp_a * blimp_a) / (blimp_c * blimp_c + blimp_a * blimp_a)) - (tuckerman_alpha - tuckerman_gamma))));
    const double tuckerman_K3_ = blimp_volume * blimp_Iz * (pow((blimp_a * blimp_a - blimp_b * blimp_b) / (blimp_a * blimp_a + blimp_b * blimp_b), 2.0) * ((tuckerman_beta - tuckerman_alpha) / (2.0 * ((blimp_a * blimp_a - blimp_b * blimp_b) / (blimp_a * blimp_a + blimp_b * blimp_b)) - (tuckerman_beta - tuckerman_alpha))));

    // c.2) Virtual masses and inertias
        // Tuckerman
    const double blimp_Xu = -tuckerman_K1 * rhoAir;
    const double blimp_Yv = -tuckerman_K2 * rhoAir;
    const double blimp_Zw = -tuckerman_K3 * rhoAir;
    const double blimp_Lp = 0.0;
    const double blimp_Mq = -tuckerman_K2_ * rhoAir;
    const double blimp_Nr = -tuckerman_K3_ * rhoAir;

    // Gomes
    const double blimp_Mu = 0.0;
    const double blimp_Lv = 0.0;
    const double blimp_Nv = 0.0;
    const double blimp_Mw = 0.0;
    const double blimp_Yp = 0.0;
    const double blimp_Xq = 0.0;
    const double blimp_Zq = 0.0;
    const double blimp_Yr = 0.0;

    // Groups
    const double blimp_mx = blimp_mTotal - blimp_Xu;
    const double blimp_my = blimp_mTotal - blimp_Yv;
    const double blimp_mz = blimp_mTotal - blimp_Zw;
    const double blimp_Jx = blimp_Ix - blimp_Lp;
    const double blimp_Jy = blimp_Iy - blimp_Mq;
    const double blimp_Jz = blimp_Iz - blimp_Nr;
    const double blimp_Jxz = 0.0;


    // d) M matrix
    printf("\n\td) M matrix");
    const double M[6][6] = {
        {blimp_mx                           , 0.0                                   , 0.0                                   , 0.0                                   , blimp_mTotal * blimp_az - blimp_Xq    , 0.0                               },
        {0.0                                , blimp_my                              , 0.0                                   , -blimp_mTotal * blimp_az - blimp_Yp   , 0.0                                   , blimp_mTotal * blimp_ax - blimp_Yr},
        {0.0                                , 0.0                                   , blimp_mz                              , 0.0                                   , -blimp_mTotal * blimp_ax - blimp_Zq   , 0.0                               },
        {0.0                                , -blimp_mTotal * blimp_az - blimp_Lv   , 0.0                                   , blimp_Ix - blimp_Lp                   , 0.0                                   , -blimp_Jxz                        },
        {blimp_mTotal * blimp_az - blimp_Mu , 0.0                                   , -blimp_mTotal * blimp_ax - blimp_Mw   , 0.0                                   , blimp_Iy - blimp_Mq                   , 0.0                               },
        {0.0                                , blimp_mTotal * blimp_ax - blimp_Nv    , 0.0                                   , -blimp_Jxz                            , 0.0                                   , blimp_Iz - blimp_Nr               }
    };

    const double invM[6][6] = {
        {0.916844157881075  , 0.0                   , 0.0               , 0.0                   , 0.287823601986896 , 0.0               },
        {0.0                , 0.666945041828523     , 0.0               , -0.638717492340345    , 0.0               , 0.0               },
        {0.0                , 0.0                   , 0.637872073637673 , 0.0                   , 0.0               , 0.0               },
        {0.0                , -0.638717492340345    , 0.0               , 14.032280169794683    , 0.0               , 0.0               },
        {0.287823601986896  , 0.0                   , 0.0               , 0.0                   , 4.489659394858919 , 0.0               },
        {0.0                , 0.0                   , 0.0               , 0.0                   , 0.0               , 4.399303334727424 }
    };

    // Step 3: Control configuration
    printf("\n\nStep 3: Control configuration...");

    // a) Time definition
    printf("\n\ta) Time definition...");
    double ti = 0.1;
    double step = 0.1;
    double tf = 600.0;
    //double tt[nData];
    for (int i = 0; i < nData; i++) {
        tt[i] = ti + step * i;
    }

    // b) Configuration
    printf("\n\tb) Configuration...");
    // b.1) DMC
    printf("\n\t\tb.1) Configuring DMC options...");
    const double alpha = 0.9;
    const double Q_par = 1.0;
    const double R1 = 2.0; // 1st Control action weight
    const double R2 = 1000.0; // 2nd Control action weight
    const double R3 = 5.0; // 3rd Control action weight
    const double Pu1 = 0000.5000; // 1st Control action proportional gain
    const double Pu2 = 0000.0000; // 2nd Control action proportional gain
    const double Pu3 = 0000.0000; // 3rd Control action proportional gain

    // Error and control action weight
    for (int i = 0; i < Np; i++) {
        for (int j = 0; j < Np; j++) {
            if (i == j) {
                Q[i][j] = Q_par;
            }
            else {
                Q[i][j] = 0.0;
            }
        }
    }
    for (int i = 0; i < Nc * mInputs; i++) {
        for (int j = 0; j < Nc * mInputs; j++) {
            if (i == j) {
                if (i < Nc) {
                    R[i][j] = R1;
                }
                else
                    if (Nc <= i && i < 2 * Nc) {
                        R[i][j] = R2;
                    }
                    else
                        if (2 * Nc <= i && i < 3 * Nc) {
                            R[i][j] = R3;
                        }
            }
            else {
                R[i][j] = 0.0;
            }
        }
    }


    // b.2) ANFIS
    printf("\n\t\tb.2) Configuring ANFIS options...");

    const double minOut = 10.0;
    const double maxOut = 90.0;

    const double minUT1 = -0.1;
    const double maxUT1 = 0.1;
    const double minUT2 = 0.45;
    const double maxUT2 = 0.55;
    const double minUT3 = -pi / 2;
    const double maxUT3 = pi / 2;

    const double minX1 = -1.0;
    const double maxX1 = 1.0;
    const double minX2 = -1.0;
    const double maxX2 = 1.0;
    const double minX3 = -1.0;
    const double maxX3 = 1.0;
    const double minX4 = -pi;
    const double maxX4 = pi;
    const double minX5 = -pi;
    const double maxX5 = pi;
    const double minX6 = -pi;
    const double maxX6 = pi;

    const double minY1 = -100;
    const double maxY1 = 100;
    const double minY2 = -100;
    const double maxY2 = 100;
    const double minY3 = -100;
    const double maxY3 = 100;
    const double minY4 = -pi;
    const double maxY4 = pi;
    const double minY5 = -pi;
    const double maxY5 = pi;
    const double minY6 = -pi;
    const double maxY6 = pi;

    // c) Workspace
    printf("\n\tc) Creating workspace...");
    // c.1) DMC
    printf("\n\t\tc.1) DMC workspace...");

    fill((double*)gi, Np, 1, 0.0);
    fill((double*)Gij, Np, mInputs * Nc, 0.0);
    fill((double*)Wref, Np, 1, 0.0);
    fill((double*)Yfree, Np, 1, 0.0);
    fill((double*)Ystep, Np, mInputs, 0.0);

    // c.2) Process
    printf("\n\t\tc.2) Process workspace...");
    fill((double*)UT, mInputs, nData, 0.0);
    fill((double*)UT + nData, 1, nData, 0.5);
    fill((double*)U, mInputs, nData, 0.0);
    fill((double*)X, mStates, nData, 0.0);
    fill((double*)Y, mStates, nData, 0.0);
    fill((double*)Vearth, 3, nData, 0.0);
    fill((double*)Xearth, 3, nData, 0.0);
    fill((double*)lambda, 3, 3, 0.0);
    fill((double*)dX, mStates, nData, 0.0);
    fill((double*)Vtot, nData, 1, 0.0);
    fill((double*)Wtot, nData, 1, 0.0);
    fill((double*)G, mStates, nData, 0.0);
    fill((double*)A, mStates, nData, 0.0);
    fill((double*)Fd, mStates, nData, 0.0);
    fill((double*)P, mStates, nData, 0.0);
    double ddu[3];

    double f1;
    double f2;
    double f3;
    double f4;
    double f5;
    double f6;

    double P1;
    double P2;
    double P3;
    double P4;
    double P5;
    double P6;

    double CD = 0.9;
    double CY = 0.9;
    double CL = 0.9;
    double Cl = 0.9;
    double Cm = 0.9;
    double Cn = 0.9;

    double coefB1;
    double coefB2;
    double coefB3;
    double coefB4;
    double coefB5;
    double coefB6;

    double A1;
    double A2;
    double A3;
    double A4;
    double A5;
    double A6;

    double G1;
    double G2;
    double G3;
    double G4;
    double G5;
    double G6;

    double aux_differential_equation[mStates];


    // c.3) ANFIS
    printf("\n\t\tc.3) ANFIS workspace...");

    double dukk[mInputs * Nc];
    fill((double*)XX, nInputs, 1, 50.0);
    fill((double*)ukk, mInputs, nData, 50.0);
    fill((double*)dukk, mInputs * Nc, 1, 0.0);
    fill((double*)xkk, mStates, nData, 50.0);
    fill((double*)ykk, mStates, nData, 50.0);
    fill((double*)yee, nData, 1, 50.0);

    // d) Reference definition
    printf("\n\td) Reference definition...");
    fill((double*)ref, nData, 1, 0.0);
    for (int i = 0; i < nData; i++) {
        ref[i] = reference(tt[i]);
    }

    // e) Reference filter
    printf("\n\te) Filtering reference...");
    fill((double*)wref, nData, 1, 0.0);
    wref[0] = ref[0];
    for (int i = 1; i < nData; i++) {
        wref[i] = alpha * wref[i - 1] + (1.0 - alpha) * ref[i];
    }

    // f) Reference plot
    printf("\n\tf) Plotting part VI...");

    // g) Index table
    printf("\n\tg) Index table...");
    printf("\n\t\td.2) Index table");
    static int indexTable[nRules][nInputs];
    for (int k = 1; k <= nInputs; k++) {
        int l = 1;
        for (int j = 1; j <= nRules; j = j + (int)pow(nFuzzy, long long(k) - 1)) {
            for (int i = 1; i <= (int)pow(nFuzzy, (long long(k) - 1)); i++) {
                indexTable[j + i - 2][nInputs - k] = l;
            }
            l = l + 1;
            if (l > nFuzzy) {
                l = 1;
            }
        }
    }
    /*
          for(int i=1; i<=nRules;i++){
                printf("\n");
            for(int j=1;j<=nInputs;j++){
                printf(" %d", indexTable[i-1][j-1]);
            }
        }
          */

          // Step 4: Control sequence
    printf("\n\nStep 4: Control sequence starts...");

    for (int n = 3; n < nData - Np; n++) {
        clock_t clock_nData1 = clock();
        // a) OUTPUT reading
        // Blimp (yk)

        clock_t clock_blimp1 = clock();
        // a) Dynamics vector, Fd
        f1 = -blimp_mz * X[2][n - 1] * X[4][n - 1] + blimp_my * X[5][n - 1] * X[1][n - 1] + blimp_mTotal * (blimp_ax * (X[4][n - 1] * X[4][n - 1] + X[5][n - 1] * X[5][n - 1]) - blimp_az * X[5][n - 1] * X[3][n - 1]);
        f2 = -blimp_mx * X[0][n - 1] * X[5][n - 1] + blimp_mz * X[3][n - 1] * X[2][n - 1] + blimp_mTotal * (-blimp_ax * X[3][n - 1] * X[4][n - 1] - blimp_az * X[5][n - 1] * X[4][n - 1]);
        f3 = -blimp_my * X[1][n - 1] * X[3][n - 1] + blimp_mx * X[4][n - 1] * X[0][n - 1] + blimp_mTotal * (-blimp_ax * X[5][n - 1] * X[3][n - 1] + blimp_az * (X[4][n - 1] * X[4][n - 1] + X[3][n - 1] * X[3][n - 1]));
        f4 = -(blimp_Jz - blimp_Jy) * X[5][n - 1] * X[4][n - 1] + blimp_Jxz * X[3][n - 1] * X[4][n - 1] + blimp_mTotal * blimp_az * (X[0][n - 1] * X[5][n - 1] - X[3][n - 1] * X[2][n - 1]);
        f5 = -(blimp_Jx - blimp_Jz) * X[3][n - 1] * X[5][n - 1] + blimp_Jxz * (X[5][n - 1] * X[5][n - 1] - X[3][n - 1] * X[3][n - 1]) + blimp_mTotal * (blimp_ax * (X[1][n - 1] * X[3][n - 1] - X[4][n - 1] * X[0][n - 1]) - blimp_az * (X[2][n - 1] * X[4][n - 1] - X[5][n - 1] * X[1][n - 1]));
        f6 = -(blimp_Jy - blimp_Jx) * X[4][n - 1] * X[3][n - 1] - blimp_Jxz * X[4][n - 1] * X[5][n - 1] + blimp_mTotal * (-blimp_ax * (X[0][n - 1] * X[5][n - 1] - X[3][n - 1] * X[2][n - 1]));
        Fd[0][n - 1] = f1;
        Fd[1][n - 1] = f2;
        Fd[2][n - 1] = f3;
        Fd[3][n - 1] = f4;
        Fd[4][n - 1] = f5;
        Fd[5][n - 1] = f6;

        // b) Propulsion vector, P
        U[0][n - 1] = UT[0][n - 1] * UT[1][n - 1];            // Alpha* Tmax
        U[1][n - 1] = UT[0][n - 1] * (1.0 - UT[1][n - 1]);      // (1 - Alpha)* Tmax
        U[2][n - 1] = UT[2][n - 1];

        P1 = (U[0][n - 1] + U[1][n - 1]) * cos(U[2][n - 1]);
        P2 = 0;
        P3 = -(U[0][n - 1] + U[1][n - 1]) * sin(U[2][n - 1]);
        P4 = (U[1][n - 1] - U[0][n - 1]) * sin(U[2][n - 1]) * blimp_dy;
        P5 = (U[0][n - 1] + U[1][n - 1]) * (blimp_dz * cos(U[2][n - 1]) - blimp_dx * sin(U[2][n - 1]));
        P6 = (U[1][n - 1] - U[0][n - 1]) * cos(U[2][n - 1]) * blimp_dy;
        P[0][n - 1] = P1;
        P[1][n - 1] = P2;
        P[2][n - 1] = P3;
        P[3][n - 1] = P4;
        P[4][n - 1] = P5;
        P[5][n - 1] = P6;

        // c) Aerodynamic force vector, A
        Vtot[n - 1] = pow(X[0][n - 1] * X[0][n - 1] + X[1][n - 1] * X[1][n - 1] + X[2][n - 1] * X[2][n - 1], 0.5);
        Wtot[n - 1] = pow(X[3][n - 1] * X[3][n - 1] + X[4][n - 1] * X[4][n - 1] + X[5][n - 1] * X[5][n - 1], 0.5);

        coefB1 = 0.5 * rhoAir * X[0][n - 1] * X[0][n - 1] * sign(X[0][n - 1]) * blimp_area;
        coefB2 = 0.5 * rhoAir * X[1][n - 1] * X[1][n - 1] * sign(X[1][n - 1]) * blimp_area;
        coefB3 = 0.5 * rhoAir * X[2][n - 1] * X[2][n - 1] * sign(X[2][n - 1]) * blimp_area;
        coefB4 = 0.5 * rhoAir * X[3][n - 1] * X[3][n - 1] * sign(X[3][n - 1]) * blimp_volume;
        coefB5 = 0.5 * rhoAir * X[4][n - 1] * X[4][n - 1] * sign(X[4][n - 1]) * blimp_volume;
        coefB6 = 0.5 * rhoAir * X[5][n - 1] * X[5][n - 1] * sign(X[5][n - 1]) * blimp_volume;

        A1 = -CD * coefB1;
        A2 = -CY * coefB2;
        A3 = -CL * coefB3;
        A4 = -Cl * coefB4;
        A5 = -Cm * coefB5;
        A6 = -Cn * coefB6;

        A[0][n - 1] = A1;
        A[1][n - 1] = A2;
        A[2][n - 1] = A3;
        A[3][n - 1] = A4;
        A[4][n - 1] = A5;
        A[5][n - 1] = A6;

        // d) Gravitational force vector, G
        lambda[0][0] = cos(Y[4][n - 1]) * cos(Y[5][n - 1]);
        lambda[0][1] = cos(Y[4][n - 1]) * sin(Y[5][n - 1]);
        lambda[0][2] = sin(Y[4][n - 1]);
        lambda[1][0] = (-cos(Y[3][n - 1]) * sin(Y[5][n - 1]) + sin(Y[3][n - 1]) * sin(Y[4][n - 1]) * cos(Y[5][n - 1]));
        lambda[1][1] = (cos(Y[3][n - 1]) * cos(Y[5][n - 1]) + sin(Y[3][n - 1]) * sin(Y[4][n - 1]) * sin(Y[5][n - 1]));
        lambda[1][2] = sin(Y[3][n - 1]) * cos(Y[4][n - 1]);
        lambda[2][0] = (sin(Y[3][n - 1]) * sin(Y[5][n - 1]) + cos(Y[3][n - 1]) * sin(Y[4][n - 1]) * cos(Y[5][n - 1]));
        lambda[2][1] = (-sin(Y[3][n - 1]) * cos(Y[5][n - 1]) + cos(Y[3][n - 1]) * sin(Y[4][n - 1]) * sin(Y[5][n - 1]));
        lambda[2][2] = cos(Y[3][n - 1]) * cos(Y[4][n - 1]);

        double B = rhoAir * g_acc * blimp_volume;
        double W = blimp_mTotal * g_acc;

        G1 = lambda[2][0] * (W - B);
        G2 = lambda[2][1] * (W - B);
        G3 = lambda[2][2] * (W - B);
        G4 = -lambda[2][1] * blimp_az * W;
        G5 = (lambda[2][0] * blimp_az - lambda[2][2] * blimp_ax) * W;
        G6 = lambda[2][1] * blimp_ax * W;

        G[0][n - 1] = G1;
        G[1][n - 1] = G2;
        G[2][n - 1] = G3;
        G[3][n - 1] = G4;
        G[4][n - 1] = G5;
        G[5][n - 1] = G6;

        // e) Differential equation
        for (int i = 0; i < mStates; i++) {
            aux_differential_equation[i] = P[i][n - 1] + Fd[i][n - 1] + A[i][n - 1] + G[i][n - 1];
        }

        for (int i = 0; i < mStates; i++) {
            for (int j = 0; j < mStates; j++) {
                dX[i][n - 1] = dX[i][n - 1] + invM[i][j] * aux_differential_equation[j];
            }
        }

        // f) Integrate differential equation
        for (int i = 0; i < mStates; i++) {
            for (int j = 0; j < n; j++) {
                X[i][n] = X[i][n] + dX[i][j];
            }
            X[i][n] = X[i][n] + (dX[i][n - 1] - dX[i][0]) * 0.5;
            X[i][n] = X[i][n] * step;
        }

        // g) Calculate vehicle position in terms of displacements in the north, east and vertical directions 
        for (int i = 0; i < mStates; i++) {
            for (int j = 0; j < n; j++) {
                Y[i][n] = Y[i][n] + X[i][j];
            }
            Y[i][n] = Y[i][n] + (X[i][n - 1] - X[i][0]) * 0.5;
            Y[i][n] = Y[i][n] * step;                           // XYZ aligned with NEU
        }


        clock_t clock_blimp2 = clock();
        // Scaling to fuzzy
        xkk[0][n] = map(X[0][n], minX1, maxX1, minOut, maxOut);
        xkk[1][n] = map(X[1][n], minX2, maxX2, minOut, maxOut);
        xkk[2][n] = map(X[2][n], minX3, maxX3, minOut, maxOut);
        xkk[3][n] = map(X[3][n], minX4, maxX4, minOut, maxOut);
        xkk[4][n] = map(X[4][n], minX5, maxX5, minOut, maxOut);
        xkk[5][n] = map(X[5][n], minX6, maxX6, minOut, maxOut);

        ykk[0][n] = map(Y[0][n], minY1, maxY1, minOut, maxOut);
        ykk[1][n] = map(Y[1][n], minY2, maxY2, minOut, maxOut);
        ykk[2][n] = map(Y[2][n], minY3, maxY3, minOut, maxOut);
        ykk[3][n] = map(Y[3][n], minY4, maxY4, minOut, maxOut);
        ykk[4][n] = map(Y[4][n], minY5, maxY5, minOut, maxOut);
        ykk[5][n] = map(Y[5][n], minY6, maxY6, minOut, maxOut);

        // b) ANFIS OUTPUT reading
        clock_t clock_ANFIS1 = clock();
        // Prelayer
        XX[0] = ukk[0][n - 1];
        XX[1] = ukk[1][n - 1];
        XX[2] = ukk[2][n - 1];
        XX[3] = xkk[0][n - 1];
        cudaStatus = ANFIS_wrapper(XX);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n\t\tANFIS_wrapper failed!");
            goto Error;
        }
        yee[n] = *host_Out;

        clock_t clock_ANFIS2 = clock();


        // Wref
        for (int j = 0; j < Np; j++) {
            Wref[j] = wref[n];
        }

        // Error
        double ekk = Wref[n] - xkk[0][n];

        // c) Prediction
        clock_t clock_prediction1 = clock();
        // ANFIS(Yfree)
        clock_t clock_Yfree1 = clock();
        Yfree[0] = xkk[0][n - 1];
        for (int z = 0; z < Np - 1; z++) {
            // Prelayer
            XX[0] = ukk[0][n - 1];
            XX[1] = ukk[1][n - 1];
            XX[2] = ukk[2][n - 1];
            XX[3] = Yfree[z];
            cudaStatus = ANFIS_wrapper(XX);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "\n\t\tANFIS_wrapper failed!");
                goto Error;
            }
            Yfree[z + 1] = *host_Out;
        }
        clock_t clock_Yfree2 = clock();
        clock_t clock_Ystep1 = clock();
        // ANFIS(Ystep)
            // ukk1
        if (ukk[0][n - 1] < 90) {
            ddu[0] = -35;
        }
        else {
            ddu[0] = +35;
        }

        // ANFIS
        for (int z = 0; z < Np; z++) {
            // Prelayer
            if (z == 0) {
                XX[3] = xkk[0][n - 1];
            }
            else {
                XX[3] = Ystep[z - 1][0];
            }
            XX[0] = ukk[0][n - 1] + ddu[0];
            XX[1] = ukk[1][n - 1];
            XX[2] = ukk[2][n - 1];
            cudaStatus = ANFIS_wrapper(XX);
            Ystep[z][0] = *host_Out;

        }



        // ukk2
        if (ukk[1][n - 1] > 50) {
            ddu[1] = -35;
        }
        else {
            ddu[1] = +35;
        }

        // ANFIS
        for (int z = 0; z < Np; z++) {
            // Prelayer
            if (z == 0) {
                XX[3] = xkk[0][n - 1];
            }
            else {
                XX[3] = Ystep[z - 1][1];
            }
            XX[0] = ukk[0][n - 1];
            XX[1] = ukk[1][n - 1] + ddu[1];
            XX[2] = ukk[2][n - 1];
            cudaStatus = ANFIS_wrapper(XX);
            Ystep[z][1] = *host_Out;
        }


        // ukk3
        if (ukk[2][n - 1] > 50) {
            ddu[2] = -35;
        }
        else {
            ddu[2] = +35;
        }

        // ANFIS
        for (int z = 0; z < Np; z++) {
            // Prelayer
            if (z == 0) {
                XX[3] = xkk[0][n - 1];
            }
            else {
                XX[3] = Ystep[z - 1][2];
            }
            XX[0] = ukk[0][n - 1];
            XX[1] = ukk[1][n - 1];
            XX[2] = ukk[2][n - 1] + ddu[2];
            cudaStatus = ANFIS_wrapper(XX);
            Ystep[z][2] = *host_Out;
        }

        clock_t clock_Ystep2 = clock();

        // Gij
        for (int nn = 0; nn < mInputs; nn++) {
            for (int j = 0; j < Np; j++) {
                gi[j] = (Ystep[j][nn] - Yfree[j]) / ddu[nn];
            }
            for (int j = 0; j < Np; j++) {
                for (int k = 0; k < Nc; k++) {
                    if (k > j) {
                        Gij[j][k + (nn)*Nc] = 0.0;
                    }
                    else {
                        Gij[j][k + (nn)*Nc] = gi[j - k];
                    }
                }
            }
        }


        // d) Optimal control sequence calculation
        clock_t clock_duk1 = clock();
        fill((double*)Gijt, mInputs * Nc, Np, 0.0);
        fill((double*)Gijt_Q, mInputs * Nc, Np, 0.0);
        fill((double*)Gijt_Q_Gij, mInputs * Nc, mInputs * Nc, 0.0);
        fill((double*)Gijt_Q_Gij_R, mInputs * Nc, mInputs * Nc, 0.0);
        fill((double*)Gijt_Q_Gij_R_1, mInputs * Nc, mInputs * Nc, 0.0);
        fill((double*)Gijt_Q_Gij_R_1_Gijt, mInputs * Nc, Np, 0.0);
        fill((double*)Gijt_Q_Gij_R_1_Gijt_Q, mInputs * Nc, Np, 0.0);
        fill((double*)Wref_Yfree, Np, 1, 0.0);
        fill((double*)dukk, mInputs * Nc, 1, 0.0);

        matrixTranspose(Np, (double*)Gij, mInputs * Nc, (double*)Gijt);
        matrixMultiplication(mInputs * Nc, (double*)Gijt, Np, (double*)Q, Np, (double*)Gijt_Q);
        matrixMultiplication(mInputs * Nc, (double*)Gijt_Q, Np, (double*)Gij, mInputs * Nc, (double*)Gijt_Q_Gij);
        matrixSum(mInputs * Nc, mInputs * Nc, (double*)Gijt_Q_Gij, (double*)R, (double*)Gijt_Q_Gij_R);
        //INV(Gijt_Q_Gij_R, Gijt_Q_Gij_R_1);
        mINVERSE((double*)Gijt_Q_Gij_R, (double*)Gijt_Q_Gij_R_1);
        matrixMultiplication(mInputs * Nc, (double*)Gijt_Q_Gij_R_1, mInputs * Nc, (double*)Gijt, Np, (double*)Gijt_Q_Gij_R_1_Gijt);
        matrixMultiplication(mInputs * Nc, (double*)Gijt_Q_Gij_R_1_Gijt, Np, (double*)Q, Np, (double*)Gijt_Q_Gij_R_1_Gijt_Q);
        matrixSubstract(Np, 1, (double*)Wref, (double*)Yfree, (double*)Wref_Yfree);
        matrixMultiplication(mInputs * Nc, (double*)Gijt_Q_Gij_R_1_Gijt_Q, Np, (double*)Wref_Yfree, 1, (double*)dukk);

        clock_t clock_duk2 = clock();

        // e) Determination of ut
        // ukk(fuzzy)
        ukk[0][n] = ukk[0][n - 1] + Pu1 * dukk[0];
        ukk[1][n] = ukk[1][n - 1] + Pu2 * dukk[Nc];
        ukk[2][n] = ukk[2][n - 1] + Pu3 * dukk[2 * Nc];

        // Limiting control action
        if (ukk[0][n] < minOut) {
            ukk[0][n] = minOut;
        }
        if (ukk[1][n] < minOut) {
            ukk[1][n] = minOut;
        }
        if (ukk[2][n] < minOut) {
            ukk[2][n] = minOut;
        }
        if (ukk[0][n] > maxOut) {
            ukk[0][n] = minOut;
        }
        if (ukk[1][n] > maxOut) {
            ukk[1][n] = minOut;
        }
        if (ukk[2][n] > maxOut) {
            ukk[2][n] = minOut;
        }



        // Scaling to Force(N)
        UT[0][n] = map(ukk[0][n], minOut, maxOut, minUT1, maxUT1);
        UT[1][n] = map(ukk[1][n], minOut, maxOut, minUT2, maxUT2);
        UT[2][n] = map(ukk[2][n], minOut, maxOut, minUT3, maxUT3);

        clock_t clock_prediction2 = clock();
        clock_t clock_nData2 = clock();
        // printf

        double clk_blimp = (double(clock_blimp2) - double(clock_blimp1)) / CLOCKS_PER_SEC;
        double clk_ANFIS = (double(clock_ANFIS2) - double(clock_ANFIS1)) / CLOCKS_PER_SEC;
        double clk_prediction = (double(clock_prediction2) - double(clock_prediction1)) / CLOCKS_PER_SEC;
        double clk_nData = (double(clock_nData2) - double(clock_nData1)) / CLOCKS_PER_SEC;
        double clk_Yfree = (double(clock_Yfree2) - double(clock_Yfree1)) / CLOCKS_PER_SEC;
        double clk_Ystep = (double(clock_Ystep2) - double(clock_Ystep1)) / CLOCKS_PER_SEC;
        double clk_duk = (double(clock_duk2) - double(clock_duk1)) / CLOCKS_PER_SEC;
        printf("\nnData = %04d - %.3fs. clk_blimp = %.3fs. clk_ANFIS = %.3fs. clk_prediction = %.3fs. clk_Yfree = %.3f s. clk_Ystep = %.3f s. clk_INV = %.3f s.", n, clk_nData, clk_blimp, clk_ANFIS, clk_prediction, clk_Yfree, clk_Ystep, clk_duk);
    }


    // Step 4: Control results
    printf("\n\nStep 4: Control results");
    printf("\n\ta) Writing to ANFISblimpCtrl.m file...");

    FILE* blimpFile;
    blimpFile = fopen("ANFISblimpCtrl.m", "w");
    if (blimpFile != NULL) {
        for (int j = 1; j <= nData; j++) {
            fprintf(blimpFile, "\nttt(%d)\t= %.15f;", j, tt[j - 1]);
        }
        for (int i = 1; i <= mInputs; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nukk(%d,%d)\t= %.15f;", i, j, ukk[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nxkk(%d,%d)\t= %.15f;", i, j, xkk[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nykk(%d,%d)\t= %.15f;", i, j, ykk[i - 1][j - 1]);
            }
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(blimpFile, "\nwref(%d)\t= %.15f;", j, wref[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(blimpFile, "\nyee(%d)\t= %.15f;", j, yee[j - 1]);
        }

        printf("\n\t¡Success!");
    }
    fclose(blimpFile);

Error:
    printf("\n\n");
    // Free allocated memory
    printf("\nFreeing allocated memory");
    free(host_In);
    free(host_Out);
    cudaFree(dev_aIn);
    cudaFree(dev_cIn);
    cudaFree(dev_aOut);
    cudaFree(dev_cOut);
    cudaFree(dev_O0);
    cudaFree(dev_O1);
    cudaFree(dev_O2);
    cudaFree(dev_O3);
    cudaFree(dev_O4);
    cudaFree(dev_O5);

    return 0;
}




//-------------------------------| 
//          FUNCTIONS            |
//-------------------------------| 

double h(double t) {
    if (t >= 0.0) {
        return 1.0;
    }
    else {
        return 0.0;
    }
}

double r(double t) {
    return h(t) * t;
}

void fill(double* arr, int n, int m, double val) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            *(arr + (long long(i) * long long(m) + long long(j))) = val;
        }
    }
}

double sign(double x) {
    if (x >= 0.0) {
        return 1.0;
    }
    else {
        return -1.0;
    }
}

double map(double input, double minIn, double maxIn, double minOut, double maxOut) {
    return ((input - minIn) / (maxIn - minIn)) * (maxOut - minOut) + minOut;
}

double reference(double t) {
    return h(t) * 50 - h(t - 30) * 10 + h(t - 250) * 10 + h(t - 350) * 10;
}

double gauss(double x, double a, double c) {
    return exp(-(x - c) * (x - c) / (a * a));
}

double sigmoid(double x, double a, double c) {
    return 1.0 / (1.0 + exp(-(x - c) / a));
}

double invSigmoid(double x, double a, double c) {
    return  c - a * log(1.0 / x - 1.0);
}

double dGauss_da(double x, double a, double c) {
    return (2.0 * exp(-(-c + x) * (-c + x) / (a * a)) * (-c + x) * (-c + x)) / (a * a * a);
}

double dGauss_dc(double x, double a, double c) {
    return (2.0 * exp(-(-c + x) * (-c + x) / (a * a)) * (-c + x)) / (a * a);
}

double dinvSigmoid_da(double x, double a, double c) {
    return -log(1.0 / x - 1.0);
}

double dinvSigmoid_dc(double x, double a, double c) {
    return 1.0;
}


void matrixMultiplication(int a, double* A, int c, double* B, int b, double* C) {
    // C = A * B
    fill((double*)C, a, b, 0.0);
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                *(C + long long(i) * long long(b) + long long(j)) = *(C + long long(i) * long long(b) + long long(j)) + *(A + long long(i) * long long(a) + long long(k)) * *(B + long long(k) * long long(b) + long long(j));
            }
        }
    }
}

void matrixTranspose(int a, double* A, int b, double* B) {
    // B = A'
    fill((double*)B, b, a, 0.0);
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            *(B + long long(i) * long long(b) + long long(j)) = *(A + long long(j) * long long(a) + long long(i));
        }
    }
}

void matrixSum(int n, int m, double* A, double* B, double* C) {
    // C = A + B
    fill((double*)C, n, m, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            *(C + long long(i) * long long(m) + long long(j)) = *(A + long long(i) * long long(m) + long long(j)) + *(B + long long(i) * long long(m) + long long(j));
        }
    }
}

void matrixSubstract(int n, int m, double* A, double* B, double* C) {
    // C = A - B
    fill((double*)C, n, m, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            *(C + long long(i) * long long(m) + long long(j)) = *(A + long long(i) * long long(m) + long long(j)) - *(B + long long(i) * long long(m) + long long(j));
        }
    }
}

void getCfactor(double M[N][N], double t[N][N], int p, int q, int n) {
    int i = 0, j = 0;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            if (r != p && c != q) {
                t[i][j++] = M[r][c];
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

double DET(double M[N][N], int n) {
    double D = 0;
    if (n == 1) {
        return M[0][0];
    }
    double t[N][N];
    int s = 1;
    for (int f = 0; f < n; f++) {
        getCfactor(M, t, 0, f, n);
        D += s * M[0][f] * DET(t, n - 1);
        s = -s;
    }
    return D;
}

void ADJ(double M[N][N], double adj[N][N]) {
    if (N == 1) {
        adj[0][0] = 1;
        return;
    }
    int s = 1;
    double t[N][N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            getCfactor(M, t, i, j, N);
            s = ((i + j) % 2 == 0) ? 1 : -1;
            adj[j][i] = (s) * (DET(t, N - 1));
        }
    }
}

bool INV(double M[N][N], double inv[N][N]) {
    double det = DET(M, N);
    if (det == 0) {
        printf("can't find its inverse");
        return false;
    }
    double adj[N][N];
    ADJ(M, adj);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            inv[i][j] = adj[i][j] / double(det);
        }
    }
    return true;
}


void mINVERSE(double M[81], double inverse[81])
{
    double x[81];
    double smax;
    int b_i;
    int i;
    int i2;
    int ix;
    int iy;
    int j;
    int jA;
    int jp1j;
    int k;
    signed char ipiv[9];
    signed char p[9];

    //  Inverse function
    for (i = 0; i < 81; i++) {
        inverse[i] = 0.0;
        x[i] = M[i];
    }

    for (i = 0; i < 9; i++) {
        ipiv[i] = static_cast<signed char>(i + 1);
    }

    for (j = 0; j < 8; j++) {
        int b_tmp;
        int mmj_tmp;
        mmj_tmp = 7 - j;
        b_tmp = j * 10;
        jp1j = b_tmp + 2;
        iy = 9 - j;
        jA = 0;
        ix = b_tmp;
        smax = fabs(x[b_tmp]);
        for (k = 2; k <= iy; k++) {
            double s;
            ix++;
            s = fabs(x[ix]);
            if (s > smax) {
                jA = k - 1;
                smax = s;
            }
        }

        if (x[b_tmp + jA] != 0.0) {
            if (jA != 0) {
                iy = j + jA;
                ipiv[j] = static_cast<signed char>(iy + 1);
                ix = j;
                for (k = 0; k < 9; k++) {
                    smax = x[ix];
                    x[ix] = x[iy];
                    x[iy] = smax;
                    ix += 9;
                    iy += 9;
                }
            }

            i = (b_tmp - j) + 9;
            for (b_i = jp1j; b_i <= i; b_i++) {
                x[b_i - 1] /= x[b_tmp];
            }
        }

        iy = b_tmp + 9;
        jA = b_tmp;
        for (jp1j = 0; jp1j <= mmj_tmp; jp1j++) {
            smax = x[iy];
            if (x[iy] != 0.0) {
                ix = b_tmp + 1;
                i = jA + 11;
                i2 = (jA - j) + 18;
                for (b_i = i; b_i <= i2; b_i++) {
                    x[b_i - 1] += x[ix] * -smax;
                    ix++;
                }
            }

            iy += 9;
            jA += 9;
        }
    }

    for (i = 0; i < 9; i++) {
        p[i] = static_cast<signed char>(i + 1);
    }

    for (k = 0; k < 8; k++) {
        signed char i1;
        i1 = ipiv[k];
        if (i1 > k + 1) {
            iy = p[i1 - 1];
            p[i1 - 1] = p[k];
            p[k] = static_cast<signed char>(iy);
        }
    }

    for (k = 0; k < 9; k++) {
        jp1j = 9 * (p[k] - 1);
        inverse[k + jp1j] = 1.0;
        for (j = k + 1; j < 10; j++) {
            i = (j + jp1j) - 1;
            if (inverse[i] != 0.0) {
                i2 = j + 1;
                for (b_i = i2; b_i < 10; b_i++) {
                    iy = (b_i + jp1j) - 1;
                    inverse[iy] -= inverse[i] * x[(b_i + 9 * (j - 1)) - 1];
                }
            }
        }
    }

    for (j = 0; j < 9; j++) {
        iy = 9 * j;
        for (k = 8; k >= 0; k--) {
            jA = 9 * k;
            i = k + iy;
            smax = inverse[i];
            if (smax != 0.0) {
                inverse[i] = smax / x[k + jA];
                for (b_i = 0; b_i < k; b_i++) {
                    jp1j = b_i + iy;
                    inverse[jp1j] -= inverse[i] * x[b_i + jA];
                }
            }
        }
    }
}


//-------------------------------| 
//        CUDA WRAPPERS          |
//-------------------------------| 

cudaError_t ANFIS_setup() {
    cudaError_t cudaStatus;

    // PARAMETERS
    cudaStatus = cudaMalloc((void**)&dev_aIn, acIn_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_aIn failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_cIn, acIn_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_cIn failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_aOut, acOut_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_aOut failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_cOut, acOut_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_cOut failed!");
        return cudaStatus;
    }

    // NODES
    cudaStatus = cudaMalloc((void**)&dev_O0, O0_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_O0 failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_O1, O1_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_O1 failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_O2, O2_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_O2 failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_O3, O3_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_O3 failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_O4, O4_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_O4 failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_O5, O5_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMalloc dev_O5 failed!");
        return cudaStatus;
    }

    // COPYING INITIAL HOST PARAMETERS TO DEVICE PARAMETERS
    printf("\n\tc) Setting up initial device (GPU) memory...");
    cudaStatus = cudaMemcpy(dev_aIn, host_aIn, acIn_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMemcpy dev_aIn failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(dev_cIn, host_cIn, acIn_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMemcpy dev_cIn failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(dev_aOut, host_aOut, acOut_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMemcpy dev_aOut failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(dev_cOut, host_cOut, acOut_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMemcpy dev_cOut failed!");
        return cudaStatus;
    }

    return cudaStatus;
}

cudaError_t ANFIS_wrapper(double In[4]) {
    cudaError_t cudaStatus;


    *(host_In) = In[0];
    *(host_In + 1) = In[1];
    *(host_In + 2) = In[2];
    *(host_In + 3) = In[3];

    // a) Send Inputs to ANFIS network
    cudaStatus = cudaMemcpy(dev_O0, host_In, 4 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n\t\tcudaMemcpy dev_O0 failed!");
        return cudaStatus;
    }

    // b) Compute output
    ANFIS <<<1, 625 >>> (dev_aIn, dev_cIn, dev_aOut, dev_cOut, dev_O0, dev_O1, dev_O2, dev_O3, dev_O4, dev_O5);

    // c) Check for errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ANFIS launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // d) Synchronize device
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }

    // e) Save output to Host (CPU)
    cudaStatus = cudaMemcpy(muIn, dev_O1, O1_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(w, dev_O2, O2_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(wn, dev_O3, O3_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(fi_wn, dev_O4, O4_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(host_Out, dev_O5, O5_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    return cudaStatus;
}




//-------------------------------| 
//             END               |
//-------------------------------| 


void set_ANFIS() {

    aIne[0][0] = 10.759043935240395484242981183343;
    aIne[1][0] = 23.375830259259604559929357492365;
    aIne[2][0] = 29.167309724297556527972119511105;
    aIne[3][0] = 21.044375442719680080472244299017;
    aIne[4][0] = 10.684510677985866067274400847964;
    aIne[0][1] = 20.915052927606399890692046028562;
    aIne[1][1] = 21.891834302520205568498568027280;
    aIne[2][1] = 29.833631863232110248418393894099;
    aIne[3][1] = 25.885131863232107463090869714506;
    aIne[4][1] = 15.511685240740394675640345667489;
    aIne[0][2] = 26.179631863232113886397200985812;
    aIne[1][2] = 26.833631863232110248418393894099;
    aIne[2][2] = 29.833631863232110248418393894099;
    aIne[3][2] = 27.333631863232110248418393894099;
    aIne[4][2] = 25.077991863232110603121327585541;
    aIne[0][3] = 21.626424658756821628458055783994;
    aIne[1][3] = 19.315682917340371460568348993547;
    aIne[2][3] = 8.087860717388554832041336339898;
    aIne[3][3] = 19.328314759259608734964785980992;
    aIne[4][3] = 23.626788471585104645100727793761;
    cIne[0][0] = -4.993948676880121340104778937530;
    cIne[1][0] = 27.004625651507542016815932583995;
    cIne[2][0] = 47.011685240740355595789878861979;
    cIne[3][0] = 78.981023482184184558718698099256;
    cIne[4][0] = 104.475682917340321864685392938554;
    cIne[0][1] = -0.044937917340372696139993990982;
    cIne[1][1] = 25.821983342111476389391100383364;
    cIne[2][1] = 50.684204887384815663153858622536;
    cIne[3][1] = 74.598100027700525060936342924833;
    cIne[4][1] = 101.338459196182128607688355259597;
    cIne[0][2] = 8.220712615289228253345754637849;
    cIne[1][2] = 33.358260512543694176201825030148;
    cIne[2][2] = 49.794519700298160103102418361232;
    cIne[3][2] = 68.674357331310829977155663073063;
    cIne[4][2] = 89.980572136767932533985003829002;
    cIne[0][3] = 9.810290080368943677058268804103;
    cIne[1][3] = 29.223282917340373643355633248575;
    cIne[2][3] = 48.071180209053999021762137999758;
    cIne[3][3] = 70.264085240740371318679535761476;
    cIne[4][3] = 90.494159338367779810141655616462;
    aOute[0] = -6.441625033716754877843868598575;
    aOute[1] = -6.118460839539333839809387427522;
    aOute[2] = -4.850547592434682364626041817246;
    aOute[3] = -5.652649421147704522638832713710;
    aOute[4] = -6.214973983919782618556837405777;
    aOute[5] = -7.721991974961774474195408402011;
    aOute[6] = -7.188268463457531787241805432132;
    aOute[7] = -6.306000668293862077007361222059;
    aOute[8] = -7.022475723699113103748459252529;
    aOute[9] = -7.522475723699113103748459252529;
    aOute[10] = -6.465915765265544123963081801776;
    aOute[11] = -4.838446803036065446690372482408;
    aOute[12] = -5.677899255091089614211341540795;
    aOute[13] = -6.580801018771056831724308722187;
    aOute[14] = -8.512499468771054011995147448033;
    aOute[15] = -2.130848115869729664240139754838;
    aOute[16] = -3.970104514759607017282405649894;
    aOute[17] = -3.569853359021361249858728115214;
    aOute[18] = -6.979873305727457122316081949975;
    aOute[19] = -8.420724817302890130576997762546;
    aOute[20] = -3.635895502588696093226872108062;
    aOute[21] = -5.284604514759607241103367414325;
    aOute[22] = -3.415099827347104710639769109548;
    aOute[23] = -6.114647342711432287387651740573;
    aOute[24] = -6.787842174210107160092775302473;
    aOute[25] = -4.663814759259608422325982246548;
    aOute[26] = -5.028314759259608024422050220892;
    aOute[27] = -3.334378991649650902928669893299;
    aOute[28] = -5.369575852678384109140097280033;
    aOute[29] = -5.258629105834977224276371998712;
    aOute[30] = -4.453480795821313620308501413092;
    aOute[31] = -5.068895720726057874117032042705;
    aOute[32] = -4.371915762163196106371287896764;
    aOute[33] = -6.018895720726057163574296282604;
    aOute[34] = -6.018895720726057163574296282604;
    aOute[35] = -3.528314759259605803976000970579;
    aOute[36] = -5.717463209259607381795831315685;
    aOute[37] = -7.947344459681517037097364664078;
    aOute[38] = -10.329158768810234647617107839324;
    aOute[39] = -10.210435889626857886014477116987;
    aOute[40] = -1.336104514759606676221892485046;
    aOute[41] = -5.478314759259607313879314460792;
    aOute[42] = -7.371779106952643267902658408275;
    aOute[43] = -9.130060095572034128963423427194;
    aOute[44] = -8.898824198144350106076672091149;
    aOute[45] = 0.013371303374241201578076498890;
    aOute[46] = -1.393068507504187092393976854510;
    aOute[47] = -2.177158123452751947723982084426;
    aOute[48] = -4.644104368033382002067810390145;
    aOute[49] = -7.694862538083381942044525203528;
    aOute[50] = -4.484724041144737327613256638870;
    aOute[51] = -5.284287695340859514203657454345;
    aOute[52] = -4.522009673452754618949711584719;
    aOute[53] = -4.621344540087213736967441946035;
    aOute[54] = -4.902341182207738512488504056819;
    aOute[55] = -2.334604514759605731200053924113;
    aOute[56] = -5.478314759259607313879314460792;
    aOute[57] = -4.706387984154670078851268044673;
    aOute[58] = -7.083336841010424933529066038318;
    aOute[59] = -7.607645562595977573039363051066;
    aOute[60] = -0.701927979141985347588672539132;
    aOute[61] = -3.478314759259605981611684910604;
    aOute[62] = -10.315238959678826091703740530647;
    aOute[63] = -11.168865327928013897462733439170;
    aOute[64] = -11.168865327928013897462733439170;
    aOute[65] = -0.425691718183896283811407101894;
    aOute[66] = -6.219852429309606023366541194264;
    aOute[67] = -10.446012640415140992899978300557;
    aOute[68] = -11.026275424395862501114606857300;
    aOute[69] = -11.031997849408270440108026377857;
    aOute[70] = 0.125782265358014982981060825296;
    aOute[71] = -5.717463209259607381795831315685;
    aOute[72] = -9.846745337682204279872166807763;
    aOute[73] = -10.445854141342810095238746725954;
    aOute[74] = -10.394616582563603657263229251839;
    aOute[75] = -2.901549897338456585060839643120;
    aOute[76] = -5.115980554868414742486493196338;
    aOute[77] = -8.321995520795468337382772006094;
    aOute[78] = -7.919724855727455548048965283670;
    aOute[79] = -8.374790943956362809785787248984;
    aOute[80] = -2.016427979141985904476541691110;
    aOute[81] = -4.978314759259607313879314460792;
    aOute[82] = -9.134382159259608613410819089040;
    aOute[83] = -8.301463209259605235956769320183;
    aOute[84] = -9.134382159259608613410819089040;
    aOute[85] = -0.063205872193553225280027163535;
    aOute[86] = -0.478314759259606314678592298151;
    aOute[87] = -10.168865327928010344749054638669;
    aOute[88] = -11.168865327928013897462733439170;
    aOute[89] = -11.168865327928013897462733439170;
    aOute[90] = -1.336102665369049535826206920319;
    aOute[91] = -7.634626641535430024987363140099;
    aOute[92] = -10.924902918312000110745429992676;
    aOute[93] = -10.750940316455086076530278660357;
    aOute[94] = -9.812632895049379300189684727229;
    aOute[95] = -3.131774357243087170132866958738;
    aOute[96] = -11.168865327928013897462733439170;
    aOute[97] = -9.551764058359907494377694092691;
    aOute[98] = -8.775492799080993222560209687799;
    aOute[99] = -8.166985838095982330742117483169;
    aOute[100] = 2.282307096430437720613326746388;
    aOute[101] = -0.578810493534755887701237497822;
    aOute[102] = -2.016427979141985904476541691110;
    aOute[103] = -2.559635837617175280200854103896;
    aOute[104] = -5.016427979141987236744171241298;
    aOute[105] = -3.535416280769128505312437482644;
    aOute[106] = -3.775107797723090907737741872552;
    aOute[107] = -7.050015756850497083974005363416;
    aOute[108] = -8.049124322142690246550955635030;
    aOute[109] = -8.601678044331674044542523915879;
    aOute[110] = -4.226097604845280741869828489143;
    aOute[111] = -4.038159766543181206088775070384;
    aOute[112] = -6.512006003936038567303512536455;
    aOute[113] = -8.049846081635152472699701320380;
    aOute[114] = -11.212204492360601904010763973929;
    aOute[115] = -0.996264819505325527337902258296;
    aOute[116] = -2.395781923659941803350648115156;
    aOute[117] = -11.513288147666592919904360314831;
    aOute[118] = -11.398904185391788956849268288352;
    aOute[119] = -9.779359800764328980449135997333;
    aOute[120] = -7.050486379912488388299607322551;
    aOute[121] = -11.168865327928013897462733439170;
    aOute[122] = -7.845889770443288391277292248560;
    aOute[123] = -7.742944204395964646892025484703;
    aOute[124] = -7.742944204395964646892025484703;
    aOute[125] = -1.791839574692096537233965136693;
    aOute[126] = -4.978314759259607313879314460792;
    aOute[127] = -5.605408050676057740702162845992;
    aOute[128] = -12.607987550289248446233614231460;
    aOute[129] = -13.847336230251350741582427872345;
    aOute[130] = -2.805490564759606453293372396729;
    aOute[131] = -4.978314759259607313879314460792;
    aOute[132] = -5.639348119183599905568371468689;
    aOute[133] = -10.487221360907383882477006409317;
    aOute[134] = -14.276903163346648995002396986820;
    aOute[135] = -0.305490564759605898181860084151;
    aOute[136] = -4.478314759259607313879314460792;
    aOute[137] = -5.518895720726057163574296282604;
    aOute[138] = -10.823022060517928366607520729303;
    aOute[139] = -14.709881283216766689747601049021;
    aOute[140] = -0.320756959759606163906653364393;
    aOute[141] = -3.952582308262317045688405414694;
    aOute[142] = -5.105408050676057740702162845992;
    aOute[143] = -12.776092600977062119227412040345;
    aOute[144] = -14.823631863232090921655981219374;
    aOute[145] = -2.707564937827851458962413744302;
    aOute[146] = -5.478314759259607313879314460792;
    aOute[147] = -5.717463209259607381795831315685;
    aOute[148] = -13.492701452443029097594262566417;
    aOute[149] = -14.823631863232090921655981219374;
    aOute[150] = -0.183307462452900943317146698064;
    aOute[151] = -4.978314759259607313879314460792;
    aOute[152] = -5.105408050676057740702162845992;
    aOute[153] = -9.599863427945457772239024052396;
    aOute[154] = -14.332995269841521945863860310055;
    aOute[155] = -1.746761624886926433930511848303;
    aOute[156] = -4.478314759259607313879314460792;
    aOute[157] = -4.478314759259607313879314460792;
    aOute[158] = -8.607426093863349336743340245448;
    aOute[159] = -14.593821105668226323359704110771;
    aOute[160] = -0.287298542827850722503058022994;
    aOute[161] = -2.978314759259605981611684910604;
    aOute[162] = -4.844556500676057808618679700885;
    aOute[163] = -9.795005463905830822568532312289;
    aOute[164] = -14.823631863232090921655981219374;
    aOute[165] = 0.654586071614741915603019606351;
    aOute[166] = -4.978314759259607313879314460792;
    aOute[167] = -4.978314759259607313879314460792;
    aOute[168] = -11.728283243613088160373081336729;
    aOute[169] = -14.823631863232090921655981219374;
    aOute[170] = 0.311491369659741224040772067383;
    aOute[171] = -4.478314759259607313879314460792;
    aOute[172] = -5.478314759259607313879314460792;
    aOute[173] = -10.010749357370549716961249941960;
    aOute[174] = -14.823631863232090921655981219374;
    aOute[175] = -2.252821314313714129440313627128;
    aOute[176] = -4.978314759259607313879314460792;
    aOute[177] = -4.263647280342009793230317882262;
    aOute[178] = -10.598616636177677463592772255652;
    aOute[179] = -14.823631863232090921655981219374;
    aOute[180] = 1.440017663646589785031437713769;
    aOute[181] = -2.978314759259605981611684910604;
    aOute[182] = -2.978314759259605981611684910604;
    aOute[183] = -9.968760907421248873561125947163;
    aOute[184] = -14.823631863232090921655981219374;
    aOute[185] = 1.556735957172148854255055994145;
    aOute[186] = -1.978314759259606425700894760666;
    aOute[187] = -4.018895720726057163574296282604;
    aOute[188] = -13.564094418777610684401224716567;
    aOute[189] = -14.823631863232090921655981219374;
    aOute[190] = 1.176372738931648953553121828008;
    aOute[191] = -5.478314759259607313879314460792;
    aOute[192] = -1.478314759259606425700894760666;
    aOute[193] = -13.069189603826877998926647705957;
    aOute[194] = -14.823631863232090921655981219374;
    aOute[195] = 3.110639239205400752297236977029;
    aOute[196] = -4.978314759259607313879314460792;
    aOute[197] = -5.478314759259607313879314460792;
    aOute[198] = -12.368170613163432136616393108852;
    aOute[199] = -14.374629259323970842388007440604;
    aOute[200] = 0.376012836346490297856348661298;
    aOute[201] = -3.261281284993718276865592997638;
    aOute[202] = -2.808809929316426146073126801639;
    aOute[203] = -10.940993695358383064331064815633;
    aOute[204] = -14.823631863232090921655981219374;
    aOute[205] = -0.236837505568351008866656570717;
    aOute[206] = -3.567340539173632585345785628306;
    aOute[207] = -0.978314759259606092633987373119;
    aOute[208] = -10.596274100304896137458854354918;
    aOute[209] = -14.823631863232090921655981219374;
    aOute[210] = 0.521685240740393907366012626881;
    aOute[211] = -0.478314759259606314678592298151;
    aOute[212] = -1.478314759259606425700894760666;
    aOute[213] = -12.423979819455347239909315248951;
    aOute[214] = -11.887521887758484950836646021344;
    aOute[215] = 2.881800539985019948829858549288;
    aOute[216] = -4.683806120158589436641705106013;
    aOute[217] = -1.478314759259606425700894760666;
    aOute[218] = -12.805760444012715737471808097325;
    aOute[219] = -14.709881283216766689747601049021;
    aOute[220] = 5.134410495274571495372128993040;
    aOute[221] = -4.978314759259607313879314460792;
    aOute[222] = -2.478314759259605981611684910604;
    aOute[223] = -11.433504611632262282228111871518;
    aOute[224] = -12.964335355583553877067970461212;
    aOute[225] = 1.009493300397343862329080366180;
    aOute[226] = -0.478314759259606314678592298151;
    aOute[227] = -0.978314759259606092633987373119;
    aOute[228] = -6.272203081854286210727877914906;
    aOute[229] = -11.520015157977390174437459791079;
    aOute[230] = 0.021685240740393751240899788968;
    aOute[231] = -0.478314759259606314678592298151;
    aOute[232] = -1.478314759259606425700894760666;
    aOute[233] = -8.343859150300914961917442269623;
    aOute[234] = -10.158429394449726146376633550972;
    aOute[235] = 1.325789847017323719313708352274;
    aOute[236] = 0.825789847017324274425220664853;
    aOute[237] = 1.805582939637106143493383569876;
    aOute[238] = -3.177785355462212191213211553986;
    aOute[239] = -9.362857733041153096564812585711;
    aOute[240] = 3.319319299403789447211465812870;
    aOute[241] = 3.041715991943478858416938237497;
    aOute[242] = 0.746683583007408002707450123125;
    aOute[243] = -5.119050016968254013249861600343;
    aOute[244] = -10.232269257891665859006025129929;
    aOute[245] = 9.152329802287376026015408569947;
    aOute[246] = 3.959283649574109720248316079960;
    aOute[247] = -0.905216350425888438557819881680;
    aOute[248] = -7.292630175229719569074404716957;
    aOute[249] = -10.067461506318760910971832345240;
    aOute[250] = 4.358030170003217129703898535809;
    aOute[251] = -4.478314759259607313879314460792;
    aOute[252] = -4.978314759259607313879314460792;
    aOute[253] = -5.478314759259607313879314460792;
    aOute[254] = -9.503340871937918876710682525299;
    aOute[255] = 4.847009603324220705644620466046;
    aOute[256] = -1.978314759259606425700894760666;
    aOute[257] = -3.605408050676056408434533295804;
    aOute[258] = -5.105408050676057740702162845992;
    aOute[259] = -8.398003908442930764977063518018;
    aOute[260] = 3.527867507650668255081427560071;
    aOute[261] = -0.978314759259606092633987373119;
    aOute[262] = -0.764719828467512408032291659765;
    aOute[263] = -2.840648963650798553004506175057;
    aOute[264] = -8.225059839578790388259221799672;
    aOute[265] = 5.915890486295091399426837597275;
    aOute[266] = -0.978314759259606092633987373119;
    aOute[267] = -1.978314759259606425700894760666;
    aOute[268] = -5.415696281299299030820293410216;
    aOute[269] = -7.815701392246112710893157782266;
    aOute[270] = 7.477555694720249945817158732098;
    aOute[271] = -1.478314759259606425700894760666;
    aOute[272] = -2.675113054092124187377521593589;
    aOute[273] = -5.105408050676057740702162845992;
    aOute[274] = -6.808286246641455541350751445862;
    aOute[275] = 6.147705246647930721337615977973;
    aOute[276] = -3.478314759259605981611684910604;
    aOute[277] = -4.463134799439718136682131444104;
    aOute[278] = -4.978314759259607313879314460792;
    aOute[279] = -6.801707510118649935293433372863;
    aOute[280] = 6.214142491454384575888525432674;
    aOute[281] = -1.978314759259606425700894760666;
    aOute[282] = -1.526263216831593716449333442142;
    aOute[283] = -4.478314759259607313879314460792;
    aOute[284] = -6.475320730001030256062222179025;
    aOute[285] = 6.823595481764887615838688361691;
    aOute[286] = -0.978314759259606092633987373119;
    aOute[287] = -1.978314759259606425700894760666;
    aOute[288] = -4.478314759259607313879314460792;
    aOute[289] = -5.949329200405232498383156780619;
    aOute[290] = 7.088502441563656830680884013418;
    aOute[291] = -1.978314759259606425700894760666;
    aOute[292] = -0.218594068116043066529741167869;
    aOute[293] = -4.978314759259607313879314460792;
    aOute[294] = -6.656666962839980605792788992403;
    aOute[295] = 7.134001008912643548853793618036;
    aOute[296] = -3.478314759259605981611684910604;
    aOute[297] = -2.096980126646973641157956080860;
    aOute[298] = -3.067314779529363821808374268585;
    aOute[299] = -3.245809221175467662590108375298;
    aOute[300] = 8.628708224694179307334707118571;
    aOute[301] = -0.478314759259606314678592298151;
    aOute[302] = -2.478314759259605981611684910604;
    aOute[303] = -0.514317082659627611285202419822;
    aOute[304] = -5.643730826491368368635903607355;
    aOute[305] = 9.167465649174207698024474666454;
    aOute[306] = 0.021685240740393751240899788968;
    aOute[307] = -1.150937214491765558932456769980;
    aOute[308] = -1.424432381904252586934944702080;
    aOute[309] = -4.328728513535302191428399964934;
    aOute[310] = 9.156860015674226360715692862868;
    aOute[311] = 0.931800539985019438127267221716;
    aOute[312] = 2.008955855831283443535539845470;
    aOute[313] = -0.991044144168716223397552766983;
    aOute[314] = -3.794788445027758694294561792049;
    aOute[315] = 8.944114950958013565696091973223;
    aOute[316] = 1.431800539985018883015754909138;
    aOute[317] = -2.300745303390467455528778373264;
    aOute[318] = -0.514317082659627611285202419822;
    aOute[319] = -3.632257151167169961780700759846;
    aOute[320] = 7.898249920719630168264302483294;
    aOute[321] = 0.998412302249483185612177749135;
    aOute[322] = -0.322647344047227058183580084005;
    aOute[323] = -0.014317082659627510671240813167;
    aOute[324] = -4.421207620052355125039866834413;
    aOute[325] = 12.326615438820503811712114838883;
    aOute[326] = 4.485682917340371389514075417537;
    aOute[327] = 0.922906320738150376392638918333;
    aOute[328] = 0.985682917340372610759402505209;
    aOute[329] = -5.629518491693884740811881783884;
    aOute[330] = 11.354831133327262904231247375719;
    aOute[331] = 4.485682917340371389514075417537;
    aOute[332] = 0.985682917340372610759402505209;
    aOute[333] = 0.985682917340372610759402505209;
    aOute[334] = -4.167003764982282199014207435539;
    aOute[335] = 10.694034337788320243589623714797;
    aOute[336] = 4.485682917340371389514075417537;
    aOute[337] = 0.535196875064880450700854908064;
    aOute[338] = 0.985682917340372610759402505209;
    aOute[339] = -3.012195047352455468114840186900;
    aOute[340] = 10.375087439558168256326098344289;
    aOute[341] = 4.485682917340371389514075417537;
    aOute[342] = -2.373380505084202063414977601497;
    aOute[343] = 0.985682917340372610759402505209;
    aOute[344] = -3.048300911468805995951925069676;
    aOute[345] = 11.603477593859814476218161871657;
    aOute[346] = 5.485682917340371389514075417537;
    aOute[347] = -1.509489874856765334598662775534;
    aOute[348] = 3.985682917340369169068026167224;
    aOute[349] = -2.828501209513806458062390447594;
    aOute[350] = 12.802293554448169743409380316734;
    aOute[351] = 5.485682917340371389514075417537;
    aOute[352] = 1.985682917340372721781704967725;
    aOute[353] = -0.038108701167169273249513139490;
    aOute[354] = -5.197061077577535215255011280533;
    aOute[355] = 13.492041166412075625657962518744;
    aOute[356] = 4.985682917340371389514075417537;
    aOute[357] = 1.485682917340372277692495117662;
    aOute[358] = 0.985682917340372610759402505209;
    aOute[359] = -4.536545609454295302498394448776;
    aOute[360] = 10.627824883640251130145770730451;
    aOute[361] = 4.485682917340371389514075417537;
    aOute[362] = 1.485682917340372277692495117662;
    aOute[363] = 0.358589625923922405981159045041;
    aOute[364] = -2.949356044538412824351780727739;
    aOute[365] = 12.049231253099584648680320242420;
    aOute[366] = 4.485682917340371389514075417537;
    aOute[367] = 1.086561922878533259506639296887;
    aOute[368] = -0.014317082659627510671240813167;
    aOute[369] = -3.591956447333780833020000500255;
    aOute[370] = 12.987555911868067681780303246342;
    aOute[371] = 5.850182917340370991610143391881;
    aOute[372] = 1.460780350484140344136108069506;
    aOute[373] = 0.494089625923922859396242301955;
    aOute[374] = -2.779413829336674357506353771896;
    aOute[375] = 9.025511957551998420967720448971;
    aOute[376] = 5.077604282154135617588508466724;
    aOute[377] = 4.493816622790785864083318301709;
    aOute[378] = 0.681578311063442687789404317300;
    aOute[379] = -2.002204783786013742030718276510;
    aOute[380] = 9.927084932385632498608174500987;
    aOute[381] = 4.356095649769833322295653488254;
    aOute[382] = 3.748468640451410482938854329404;
    aOute[383] = 2.996551550440500477634486742318;
    aOute[384] = -2.690983879978465775906215640134;
    aOute[385] = 10.425249485876729949040964129381;
    aOute[386] = 1.819771973410098109269483757089;
    aOute[387] = 2.179999405284883451372479612473;
    aOute[388] = -2.002204783786013742030718276510;
    aOute[389] = -2.353557563528953178177971494733;
    aOute[390] = 14.843631863232090495330339763314;
    aOute[391] = 8.123448651389917785081706824712;
    aOute[392] = 3.726784562220082808181587097351;
    aOute[393] = 0.985682917340372610759402505209;
    aOute[394] = -0.514317082659627611285202419822;
    aOute[395] = 14.843631863232090495330339763314;
    aOute[396] = 8.547144421176025019804001203738;
    aOute[397] = 4.957101367271479652742982580094;
    aOute[398] = 1.485682917340372277692495117662;
    aOute[399] = 0.985682917340372610759402505209;
    aOute[400] = 9.301392702509748033889991347678;
    aOute[401] = 7.283186617786261862761421070900;
    aOute[402] = 4.619083566050005806857825518819;
    aOute[403] = 1.485682917340372277692495117662;
    aOute[404] = -0.014317082659627510671240813167;
    aOute[405] = 10.041379773549355292061591171660;
    aOute[406] = 7.967173990292934604440233670175;
    aOute[407] = 1.208968266805379609962756148889;
    aOute[408] = 1.485682917340372277692495117662;
    aOute[409] = -0.491044144168716334419855229498;
    aOute[410] = 12.205089966611122775930198258720;
    aOute[411] = 10.051480906048645636019500670955;
    aOute[412] = 3.042096261402339241186609797296;
    aOute[413] = 0.053225503224022141512961070475;
    aOute[414] = -0.014317082659627510671240813167;
    aOute[415] = 14.843631863232090495330339763314;
    aOute[416] = 8.659661383060026196289982181042;
    aOute[417] = 2.508737959251964966256309708115;
    aOute[418] = 1.485682917340372277692495117662;
    aOute[419] = 0.485682917340372888315158661499;
    aOute[420] = 14.843631863232090495330339763314;
    aOute[421] = 8.306948705780307307122711790726;
    aOute[422] = 4.786613215340504545736166619463;
    aOute[423] = 2.485682917340372721781704967725;
    aOute[424] = 2.485682917340372721781704967725;
    aOute[425] = 14.744296996597629600955770001747;
    aOute[426] = 7.514071190170898972837676410563;
    aOute[427] = 4.927827983581004822610793780768;
    aOute[428] = 3.485682917340372721781704967725;
    aOute[429] = 2.485682917340372721781704967725;
    aOute[430] = 14.843631863232090495330339763314;
    aOute[431] = 9.639421189764668440602690679953;
    aOute[432] = 4.890372826369811853908231569221;
    aOute[433] = 2.985682917340372721781704967725;
    aOute[434] = 2.485682917340372721781704967725;
    aOute[435] = 14.843631863232090495330339763314;
    aOute[436] = 9.295103325676791428122669458389;
    aOute[437] = 5.073572626565467480475035699783;
    aOute[438] = 2.485682917340372721781704967725;
    aOute[439] = 0.985682917340372610759402505209;
    aOute[440] = 14.843631863232090495330339763314;
    aOute[441] = 10.441611464591790792155734379776;
    aOute[442] = 6.364036635511981288004790258128;
    aOute[443] = 4.485682917340371389514075417537;
    aOute[444] = 3.485682917340372721781704967725;
    aOute[445] = 14.843631863232090495330339763314;
    aOute[446] = 9.283387663327303584992478135973;
    aOute[447] = 6.433199147047321986292445217259;
    aOute[448] = 4.985682917340371389514075417537;
    aOute[449] = 4.485682917340371389514075417537;
    aOute[450] = 14.479131863232090893234271788970;
    aOute[451] = 7.715689648459903793309422326274;
    aOute[452] = 5.526263878806821239209057239350;
    aOute[453] = 3.905221675366456590694497208460;
    aOute[454] = 3.985682917340369169068026167224;
    aOute[455] = 14.843631863232090495330339763314;
    aOute[456] = 9.577274616642778681807612883858;
    aOute[457] = 4.118914888249414651966162637109;
    aOute[458] = 3.985682917340369169068026167224;
    aOute[459] = 3.485682917340372721781704967725;
    aOute[460] = 14.843631863232090495330339763314;
    aOute[461] = 9.515518576875194867170648649335;
    aOute[462] = 4.884635388249413345818084053462;
    aOute[463] = 3.485682917340372721781704967725;
    aOute[464] = 3.485682917340372721781704967725;
    aOute[465] = 14.843631863232090495330339763314;
    aOute[466] = 9.614782959165857789685105672106;
    aOute[467] = 5.526263878806821239209057239350;
    aOute[468] = 4.021575325783244281296902045142;
    aOute[469] = 4.985682917340371389514075417537;
    aOute[470] = 13.531545583843637814425164833665;
    aOute[471] = 9.406645077598067672397519345395;
    aOute[472] = 6.291984378806821709417818055954;
    aOute[473] = 3.395884979266979186718344863039;
    aOute[474] = 5.485682917340371389514075417537;
    aOute[475] = 11.578403693054500678272233926691;
    aOute[476] = 6.369999123838837817856983747333;
    aOute[477] = 5.224831367340371457430592272431;
    aOute[478] = 4.485682917340371389514075417537;
    aOute[479] = 4.485682917340371389514075417537;
    aOute[480] = 14.843631863232090495330339763314;
    aOute[481] = 8.748054175469643922724571893923;
    aOute[482] = 5.884635388249414233996503753588;
    aOute[483] = 3.985682917340369169068026167224;
    aOute[484] = 3.485682917340372721781704967725;
    aOute[485] = 14.843631863232090495330339763314;
    aOute[486] = 10.033799592818972712393588153645;
    aOute[487] = 5.852075297252969043881876132218;
    aOute[488] = 3.485682917340372721781704967725;
    aOute[489] = 3.485682917340372721781704967725;
    aOute[490] = 14.843631863232090495330339763314;
    aOute[491] = 7.896762556858183401686801516917;
    aOute[492] = 5.724831367340371457430592272431;
    aOute[493] = 5.485682917340371389514075417537;
    aOute[494] = 5.485682917340371389514075417537;
    aOute[495] = 13.095106209388866602694179164246;
    aOute[496] = 8.790847028694024345440993783996;
    aOute[497] = 6.664891087390371282594969670754;
    aOute[498] = 5.771752600536601107705791946501;
    aOute[499] = 6.112776208756821816336923802737;
    aOute[500] = 7.335180831975065984806860797107;
    aOute[501] = 5.796341611925064363219917140668;
    aOute[502] = 7.229306828193646339286715374328;
    aOute[503] = 0.808386108129805047717297838972;
    aOute[504] = -2.423270200796598050629881981877;
    aOute[505] = 11.121555312368855084059759974480;
    aOute[506] = 10.590686148900875451772662927397;
    aOute[507] = 11.381733128674735411323126754723;
    aOute[508] = 2.492179902420677883867483615177;
    aOute[509] = -4.728822482531268356353848503204;
    aOute[510] = 14.699451555443388883759325835854;
    aOute[511] = 9.368869528395300960710301296785;
    aOute[512] = 9.843631863232079837189303361811;
    aOute[513] = -0.863292219575693242639147229056;
    aOute[514] = -4.400053225203287787792305607582;
    aOute[515] = 14.509049125137135405338995042257;
    aOute[516] = 10.341871931220424585262662731111;
    aOute[517] = 10.343631863232079837189303361811;
    aOute[518] = 1.485682917340372277692495117662;
    aOute[519] = -2.841255364049684484228919245652;
    aOute[520] = 11.562473313585098111389015684836;
    aOute[521] = 10.277532110852991564797775936313;
    aOute[522] = 11.160392487715100173772952985018;
    aOute[523] = 1.985682917340372721781704967725;
    aOute[524] = 0.985682917340372610759402505209;
    aOute[525] = 4.422125187875828089545393595472;
    aOute[526] = 3.747785967825828290500567163690;
    aOute[527] = 3.717370187875828957402291052858;
    aOute[528] = 1.431399285364187257130197394872;
    aOute[529] = 0.198628722347956726546058803251;
    aOute[530] = 10.469698192046529072740668198094;
    aOute[531] = 9.579023452670426763688737992197;
    aOute[532] = 10.997829231416620388017690856941;
    aOute[533] = 2.990551867340373259906982639222;
    aOute[534] = 1.724831367340372345609011972556;
    aOute[535] = 14.843631863232090495330339763314;
    aOute[536] = 10.843631863232083389902982162312;
    aOute[537] = 9.843631863232079837189303361811;
    aOute[538] = 1.224831367340372345609011972556;
    aOute[539] = 0.985682917340372610759402505209;
    aOute[540] = 13.343631863232083389902982162312;
    aOute[541] = 11.624131863232079808767593931407;
    aOute[542] = 11.893631863232076995018360321410;
    aOute[543] = 2.553225503224022308046414764249;
    aOute[544] = 0.985682917340372610759402505209;
    aOute[545] = 11.060306496191884662039228715003;
    aOute[546] = 9.997801775676794733271890436299;
    aOute[547] = 9.928730469446229278673854423687;
    aOute[548] = 3.960072280315114934268194701872;
    aOute[549] = 2.720923830315114866351677846978;
    aOute[550] = 11.578865033216093394230483681895;
    aOute[551] = 9.549384495451366916540791862644;
    aOute[552] = 11.957042560221967875122572877444;
    aOute[553] = 4.485682917340371389514075417537;
    aOute[554] = 3.985682917340369169068026167224;
    aOute[555] = 12.381974731738129236191525706090;
    aOute[556] = 12.024585146105618349565702374093;
    aOute[557] = 12.849535767778892392243506037630;
    aOute[558] = 4.485682917340371389514075417537;
    aOute[559] = 2.723743868897501752002199282288;
    aOute[560] = 13.127354452090020942023329553194;
    aOute[561] = 10.751610750435101948596638976596;
    aOute[562] = 12.252864786982932798764522885904;
    aOute[563] = 3.485682917340372721781704967725;
    aOute[564] = 0.862013816891271544839980833785;
    aOute[565] = 11.686683954753867453746352111921;
    aOute[566] = 7.063588297738117915969269233756;
    aOute[567] = 10.821832738308074794986168853939;
    aOute[568] = 4.485682917340371389514075417537;
    aOute[569] = 2.485682917340372721781704967725;
    aOute[570] = 9.990340845246567624826639075764;
    aOute[571] = 7.376059277438963945883187989239;
    aOute[572] = 9.737127731989261292255832813680;
    aOute[573] = 4.485682917340371389514075417537;
    aOute[574] = 2.985682917340372721781704967725;
    aOute[575] = 11.708882991156961850265361135826;
    aOute[576] = 11.578865033216093394230483681895;
    aOute[577] = 11.833373459772865032846311805770;
    aOute[578] = 4.947569697457991466649218637031;
    aOute[579] = 4.947569697457991466649218637031;
    aOute[580] = 12.092127731989268824008831870742;
    aOute[581] = 11.957042560221967875122572877444;
    aOute[582] = 12.529899749081909732240092125721;
    aOute[583] = 4.771752600536601107705791946501;
    aOute[584] = 3.088571707313973568886922294041;
    aOute[585] = 12.381974731738129236191525706090;
    aOute[586] = 9.585340845246570040671940660104;
    aOute[587] = 11.972809930938229427965779905207;
    aOute[588] = 4.362013816891270323594653746113;
    aOute[589] = 2.359158401716042963158770362497;
    aOute[590] = 10.745737455989921471655179630034;
    aOute[591] = 7.682390158756820852659075171687;
    aOute[592] = 12.307717088188564247275280649774;
    aOute[593] = 4.985682917340371389514075417537;
    aOute[594] = 3.947569697457989246203169386717;
    aOute[595] = 10.892234896547332567706689587794;
    aOute[596] = 10.143822213756831018827142543159;
    aOute[597] = 11.359120732357052219185788999312;
    aOute[598] = 5.485682917340371389514075417537;
    aOute[599] = 4.985682917340371389514075417537;
    aOute[600] = 11.689424731738128571123525034636;
    aOute[601] = 9.810179731738120523232282721438;
    aOute[602] = 10.792166490015350888143075280823;
    aOute[603] = 6.195659903788838462901367165614;
    aOute[604] = 5.205427218199414340915609500371;
    aOute[605] = 13.221051864033338318904498009942;
    aOute[606] = 13.148961710138987513118991046213;
    aOute[607] = 13.097453807021565808099694550037;
    aOute[608] = 5.939820238076663372339680790901;
    aOute[609] = 4.578333926782963914092761115171;
    aOute[610] = 14.843631863232090495330339763314;
    aOute[611] = 14.343631863232086942616660962813;
    aOute[612] = 14.153940242866990928405357408337;
    aOute[613] = 6.387467286235120056403502530884;
    aOute[614] = 4.944575668199414408832126355264;
    aOute[615] = 12.259631863232080206671525957063;
    aOute[616] = 11.567081863232079541603525285609;
    aOute[617] = 12.624131863232079808767593931407;
    aOute[618] = 5.630907142745071425338210246991;
    aOute[619] = 4.705427218199414340915609500371;
    aOute[620] = 11.451771741799642967407635296695;
    aOute[621] = 11.451771741799642967407635296695;
    aOute[622] = 11.789344233130876204995729494840;
    aOute[623] = 6.617645158756822354462201474234;
    aOute[624] = 6.351924658756821884253440657631;
    cOute[0] = 6.565375613732077120232588640647;
    cOute[1] = 6.725632023988489471832963317866;
    cOute[2] = 5.670729981683501286227055970812;
    cOute[3] = 6.663656185148431276843439263757;
    cOute[4] = 7.269486480564975039442288107239;
    cOute[5] = 9.108249964514129715098533779383;
    cOute[6] = 9.014927403946686013114231172949;
    cOute[7] = 7.370263703979057723358891962562;
    cOute[8] = 8.105730503034735079381789546460;
    cOute[9] = 9.209328219861987463445984758437;
    cOute[10] = 8.896939716296181188681657658890;
    cOute[11] = 9.364924372650051509481272660196;
    cOute[12] = 7.945317316855458500413078581914;
    cOute[13] = 8.815349120344887268174716155045;
    cOute[14] = 11.987524480601301490878540789708;
    cOute[15] = 6.176419786898629027405149827246;
    cOute[16] = 8.044148138254231028554386284668;
    cOute[17] = 8.948484715656046972753756563179;
    cOute[18] = 11.449081276226490189174000988714;
    cOute[19] = 11.906295100599248115713635343127;
    cOute[20] = 8.057810980543314016699696367141;
    cOute[21] = 9.032018781749950875337162869982;
    cOute[22] = 8.746454238708963657700223848224;
    cOute[23] = 10.316141240519808519593425444327;
    cOute[24] = 11.545393774748923476636264240369;
    cOute[25] = 8.829569177790379086445682332851;
    cOute[26] = 9.204981425926270333093270892277;
    cOute[27] = 7.910255561361218212823587236926;
    cOute[28] = 9.975494246439090773037605686113;
    cOute[29] = 9.974458027925294345550355501473;
    cOute[30] = 10.116598014629360591243312228471;
    cOute[31] = 11.078723005812491919641615822911;
    cOute[32] = 9.398522154880552648137381765991;
    cOute[33] = 11.474262557232595582945577916689;
    cOute[34] = 11.634518967489004381832273793407;
    cOute[35] = 8.685402338116343656793105765246;
    cOute[36] = 11.671033198540374797858021338470;
    cOute[37] = 14.274044713160044750566157745197;
    cOute[38] = 16.254087372875584804887694190256;
    cOute[39] = 16.717828863210986156673243385740;
    cOute[40] = 7.794474144898397405256673664553;
    cOute[41] = 11.608827579772427185389460646547;
    cOute[42] = 14.468996710962365170871635200456;
    cOute[43] = 15.733651350745429198241254198365;
    cOute[44] = 16.444066378742352441122420714237;
    cOute[45] = 7.300818167606813346992566948757;
    cOute[46] = 8.088621250772131787698526750319;
    cOute[47] = 9.719209405504035004241814021952;
    cOute[48] = 11.941412060341072276514751138166;
    cOute[49] = 15.557426640647470605927082942799;
    cOute[50] = 12.269504515705392577729071490467;
    cOute[51] = 12.211925662464206965296398266219;
    cOute[52] = 12.939206423953176994245950481854;
    cOute[53] = 13.099462834209585793132646358572;
    cOute[54] = 13.259719244466001697446699836291;
    cOute[55] = 9.728671848744550487708693253808;
    cOute[56] = 14.268963489118579701653288793750;
    cOute[57] = 14.875603505275442728361667832360;
    cOute[58] = 15.822600246437421134260148392059;
    cOute[59] = 16.076379141777440651139841065742;
    cOute[60] = 9.931891886382459233573172241449;
    cOute[61] = 11.352069004783015060411344165914;
    cOute[62] = 20.547212875358713546347644296475;
    cOute[63] = 21.275019174081865713787919958122;
    cOute[64] = 21.435275584338274512674615834840;
    cOute[65] = 10.777346730414777198348019737750;
    cOute[66] = 15.516838726115068425315257627517;
    cOute[67] = 21.968608616511065889653764315881;
    cOute[68] = 22.278134141570351545169614837505;
    cOute[69] = 22.214074119375393223663195385598;
    cOute[70] = 9.355501434734740939802577486262;
    cOute[71] = 14.416519887464733784554482554086;
    cOute[72] = 21.943078121025148874423393863253;
    cOute[73] = 22.281665381477214538108455599286;
    cOute[74] = 22.247784296971740758408486726694;
    cOute[75] = 12.302059391519225783895308268256;
    cOute[76] = 16.837053970444042505505422013812;
    cOute[77] = 21.028731918196488237526864395477;
    cOute[78] = 20.317592493956379229302910971455;
    cOute[79] = 21.483717815788221372486077598296;
    cOute[80] = 13.427056383715299858749858685769;
    cOute[81] = 18.007197209911215196598277543671;
    cOute[82] = 23.119492020335254522933610132895;
    cOute[83] = 22.445664210541668381893032346852;
    cOute[84] = 23.265665620798078094821903505363;
    cOute[85] = 14.966917808765426656236741109751;
    cOute[86] = 16.066231775199753428751137107611;
    cOute[87] = 22.774118586283034204598152427934;
    cOute[88] = 25.281429430492121213092104881071;
    cOute[89] = 25.544631406795854644542487221770;
    cOute[90] = 15.868477139318478918994514970109;
    cOute[91] = 19.296636831410491907945470302366;
    cOute[92] = 24.389133986061853676119426381774;
    cOute[93] = 24.497330996815538384225874324329;
    cOute[94] = 24.344703937621904543675555032678;
    cOute[95] = 17.534394077230309960668819257990;
    cOute[96] = 26.563480712543398709613029495813;
    cOute[97] = 23.620505866174095643827968160622;
    cOute[98] = 23.780762276430504442714664037339;
    cOute[99] = 23.941018686686913241601359914057;
    cOute[100] = 12.601902618794067478802389814518;
    cOute[101] = 16.615949033967002179679184337147;
    cOute[102] = 18.372581825295821289500963757746;
    cOute[103] = 19.118394116118960113226421526633;
    cOute[104] = 21.236302504283838032961284625344;
    cOute[105] = 21.403120496616114820653820061125;
    cOute[106] = 23.069614190428143984945563715883;
    cOute[107] = 25.827920451022546188823980628513;
    cOute[108] = 26.613740977921491293045619386248;
    cOute[109] = 28.178997388177901228800692479126;
    cOute[110] = 22.090317609990048453028066433035;
    cOute[111] = 23.105201831857687722049377043732;
    cOute[112] = 25.798793883675596561033671605401;
    cOute[113] = 27.806819290468201444355145213194;
    cOute[114] = 30.078365853948678676488270866685;
    cOute[115] = 19.385269103147120262065072893165;
    cOute[116] = 20.495525513403531903122711810283;
    cOute[117] = 29.805462734372834887608405551873;
    cOute[118] = 29.677647619522357302912496379577;
    cOute[119] = 27.287741527782184647321628290229;
    cOute[120] = 27.061313450178612782792697544210;
    cOute[121] = 30.535589071059515475781154236756;
    cOute[122] = 27.304226255677985335523771937005;
    cOute[123] = 27.464482665934394134410467813723;
    cOute[124] = 27.624739076190802933297163690440;
    cOute[125] = 18.692746383880862737214556545950;
    cOute[126] = 24.064166382939323085565774817951;
    cOute[127] = 32.506842704820272160759486723691;
    cOute[128] = 35.032591752325522804767388151959;
    cOute[129] = 35.340676697538704331691405968741;
    cOute[130] = 23.648823898092931727887844317593;
    cOute[131] = 25.788194258349346199565843562596;
    cOute[132] = 32.148826048019508050401782384142;
    cOute[133] = 36.033983847319298376987717347220;
    cOute[134] = 36.307990837591027855069114593789;
    cOute[135] = 20.852555949374984578525982215069;
    cOute[136] = 25.283186554131393819488948793150;
    cOute[137] = 31.384619820225093889121126267128;
    cOute[138] = 36.949016478616663050615898100659;
    cOute[139] = 37.109272888873071849502593977377;
    cOute[140] = 22.577368447273638452088562189601;
    cOute[141] = 27.125037860913447929078756715171;
    cOute[142] = 33.978966712514512948928313562647;
    cOute[143] = 37.750298529898721255904092686251;
    cOute[144] = 37.910554940155130054790788562968;
    cOute[145] = 24.437655666981232371881560538895;
    cOute[146] = 28.692040412195499499148354516365;
    cOute[147] = 35.787386884376452655942557612434;
    cOute[148] = 38.551580581180765250337572069839;
    cOute[149] = 38.711836991437188260078983148560;
    cOute[150] = 20.875931368458285675160368555225;
    cOute[151] = 26.359922364056526333797592087649;
    cOute[152] = 33.359825804880934185803198488429;
    cOute[153] = 38.506062052447497023877076571807;
    cOute[154] = 39.513119042719246465367177734151;
    cOute[155] = 23.786372875441816887587265227921;
    cOute[156] = 29.018733811068347705486303311773;
    cOute[157] = 34.270755169100745263222052017227;
    cOute[158] = 39.249144683744880524045584024861;
    cOute[159] = 40.314401094001304670655372319743;
    cOute[160] = 24.339534271365888429272672510706;
    cOute[161] = 30.095886566041659904158223071136;
    cOute[162] = 32.902675967642714738303766353056;
    cOute[163] = 40.955426735026939866202155826613;
    cOute[164] = 41.115683145283362875943566905335;
    cOute[165] = 25.312766322647942018875255598687;
    cOute[166] = 30.405575325907260975100143696181;
    cOute[167] = 35.551395116738440549397637369111;
    cOute[168] = 41.756708786308998071490350412205;
    cOute[169] = 41.916965196565406870377046288922;
    cOute[170] = 26.335065889941741801294483593665;
    cOute[171] = 29.139953455471907517448926228099;
    cOute[172] = 34.691794775734798861321905860677;
    cOute[173] = 42.557990837591056276778544997796;
    cOute[174] = 42.718247247847465075665240874514;
    cOute[175] = 29.084841897259625653759940178134;
    cOute[176] = 32.221480227844587318486446747556;
    cOute[177] = 37.540324952863507235178985865787;
    cOute[178] = 43.359272888873100271212024381384;
    cOute[179] = 43.519529299129509070098720258102;
    cOute[180] = 23.959667508999711316164393792860;
    cOute[181] = 30.476296287712262511604421888478;
    cOute[182] = 36.744077274580114078617043560371;
    cOute[183] = 44.160554940155158476500218966976;
    cOute[184] = 44.320811350411567275386914843693;
    cOute[185] = 26.628756317168740253009673324414;
    cOute[186] = 30.475203531035472082066917209886;
    cOute[187] = 33.449453831309234885793557623401;
    cOute[188] = 44.961836991437202470933698350564;
    cOute[189] = 45.122093401693611269820394227281;
    cOute[190] = 26.955111193500801647360276547261;
    cOute[191] = 30.647289118233967286641927785240;
    cOute[192] = 41.031695738001864981470134807751;
    cOute[193] = 45.763119042719260676221892936155;
    cOute[194] = 45.923375452975669475108588812873;
    cOute[195] = 26.673801893469107682221874711104;
    cOute[196] = 30.905226687098330984326821635477;
    cOute[197] = 38.662476029772442132070864317939;
    cOute[198] = 45.823146792747429856262897374108;
    cOute[199] = 46.724657504257741891251498600468;
    cOute[200] = 30.752386864454191339746103039943;
    cOute[201] = 35.072759929381668086989520816132;
    cOute[202] = 40.012619537712275530338956741616;
    cOute[203] = 47.365683145283391297652997309342;
    cOute[204] = 47.525939555539800096539693186060;
    cOute[205] = 31.726638061249651912021363386884;
    cOute[206] = 35.001135272080155402818490983918;
    cOute[207] = 40.761827681177059901074244407937;
    cOute[208] = 48.166965196565449502941191894934;
    cOute[209] = 48.327221606821858301827887771651;
    cOute[210] = 32.165433851596695546959381317720;
    cOute[211] = 32.892302024117576308981369948015;
    cOute[212] = 33.651904160174574087704968405887;
    cOute[213] = 48.968247247847493497374671278521;
    cOute[214] = 46.568403680400599853328458266333;
    cOute[215] = 30.813827665143211476106444024481;
    cOute[216] = 33.626972313135162551134271780029;
    cOute[217] = 43.087104234900685639786388492212;
    cOute[218] = 49.769529299129551702662865864113;
    cOute[219] = 49.929785709385960501549561740831;
    cOute[220] = 28.515277239838152212314525968395;
    cOute[221] = 36.676821488936589332752191694453;
    cOute[222] = 43.649237836182742000801226822659;
    cOute[223] = 48.272774610935051953219954157248;
    cOute[224] = 49.628108372476212650781235424802;
    cOute[225] = 32.711146939092451191299915080890;
    cOute[226] = 36.706263477208359802261838922277;
    cOute[227] = 37.493613178881211922544025583193;
    cOute[228] = 43.647337152193635745334177045152;
    cOute[229] = 49.016679031019343426578416256234;
    cOute[230] = 36.847289118234009208663337631151;
    cOute[231] = 37.507545528490418007550033507869;
    cOute[232] = 38.794895230163270127832220168784;
    cOute[233] = 47.632227317960364132432005135342;
    cOute[234] = 48.006361248141182329618459334597;
    cOute[235] = 36.344466563239116396744066150859;
    cOute[236] = 36.320939878646065324119263095781;
    cOute[237] = 36.416086293159118270068574929610;
    cOute[238] = 42.258931082556181024756369879469;
    cOute[239] = 48.170916237914120472396461991593;
    cOute[240] = 34.929822469595016798393771750852;
    cOute[241] = 35.204101148997416714792052516714;
    cOute[242] = 38.864357559253825513678748393431;
    cOute[243] = 47.411695547420727336884738178924;
    cOute[244] = 49.882313588440815976809972198680;
    cOute[245] = 29.501567108331183675318243331276;
    cOute[246] = 31.283337969365273778521441272460;
    cOute[247] = 40.209825045192310710717720212415;
    cOute[248] = 48.982904937448971338653791463003;
    cOute[249] = 51.103066806307090530481218593195;
    cOute[250] = 35.508968377470516486482665641233;
    cOute[251] = 41.248952623395290117969125276431;
    cOute[252] = 49.482936321717453154178656404838;
    cOute[253] = 52.714987731973884876879310468212;
    cOute[254] = 52.934822781812407299639744451270;
    cOute[255] = 33.101374972662078732810186920688;
    cOute[256] = 40.096606427502074154745059786364;
    cOute[257] = 46.255580634678487683686398668215;
    cOute[258] = 50.309917600331353071396733867005;
    cOute[259] = 51.189627907023300679156818659976;
    cOute[260] = 35.658121086170361024869635002688;
    cOute[261] = 42.315237836182724606715055415407;
    cOute[262] = 46.953323525440303853883960982785;
    cOute[263] = 49.154030501809081954434077488258;
    cOute[264] = 51.899847219530023778588656568900;
    cOute[265] = 35.500393023228532740631635533646;
    cOute[266] = 38.241857350725872777275071712211;
    cOute[267] = 45.247781920645714137663162546232;
    cOute[268] = 52.228099002676181328297388972715;
    cOute[269] = 52.403246029707027275890141027048;
    cOute[270] = 33.177982269357592315373040037230;
    cOute[271] = 36.748208260626221033362526213750;
    cOute[272] = 49.350532112942708806713199010119;
    cOute[273] = 56.770166435949555250317644095048;
    cOute[274] = 54.084890755300868647736933780834;
    cOute[275] = 36.700878633880755330665124347433;
    cOute[276] = 42.766560240597605968559946632013;
    cOute[277] = 50.611588103819492800994339631870;
    cOute[278] = 52.834668686682284999278635950759;
    cOute[279] = 53.915341576029668146929907379672;
    cOute[280] = 36.554442193917786596557562006637;
    cOute[281] = 41.802836410432611558007920393720;
    cOute[282] = 50.845338318991871062735299346969;
    cOute[283] = 53.508857446547885672316624550149;
    cOute[284] = 54.258401429067653509719093563035;
    cOute[285] = 37.439183718865969296984985703602;
    cOute[286] = 40.017594425798215240774879930541;
    cOute[287] = 48.748375246638453006653435295448;
    cOute[288] = 53.806792510050136968402512138709;
    cOute[289] = 54.606495134200180530115176225081;
    cOute[290] = 36.956611672926293010732479160652;
    cOute[291] = 38.345513182795592399543238570914;
    cOute[292] = 48.955370477835835174573730910197;
    cOute[293] = 57.606495532584084173777227988467;
    cOute[294] = 57.316327723202611821307073114440;
    cOute[295] = 37.906006839008959730108472285792;
    cOute[296] = 40.980732224218293424655712442473;
    cOute[297] = 49.690548650331344049391191219911;
    cOute[298] = 56.197952302321631634640652919188;
    cOute[299] = 54.211330827722953529246296966448;
    cOute[300] = 38.207288890291017935396666871384;
    cOute[301] = 41.452267660295568418860057136044;
    cOute[302] = 53.488235003650622445547924144194;
    cOute[303] = 56.599951650256294044538663001731;
    cOute[304] = 56.725122635669741555375367170200;
    cOute[305] = 38.150737151083397691309073707089;
    cOute[306] = 40.062420025477472051989025203511;
    cOute[307] = 52.225570167011817090951808495447;
    cOute[308] = 58.794759220082134731910628033802;
    cOute[309] = 57.352851216130936506942816777155;
    cOute[310] = 38.658426288367188305983290774748;
    cOute[311] = 39.513032671108483384614373790100;
    cOute[312] = 49.891998743720726849915081402287;
    cOute[313] = 57.454409541608590927808108972386;
    cOute[314] = 55.917958351108289605235768249258;
    cOute[315] = 40.006355804900415762404009001330;
    cOute[316] = 40.433768845214267173560074297711;
    cOute[317] = 49.325599133941658180901868036017;
    cOute[318] = 58.822984995588548429168440634385;
    cOute[319] = 58.363817553004487592716031940654;
    cOute[320] = 41.619521112048097677416080841795;
    cOute[321] = 41.925219997992307696677016792819;
    cOute[322] = 48.387732735223700331061991164461;
    cOute[323] = 59.072385779913766157278587343171;
    cOute[324] = 59.471790640170176800438639475033;
    cOute[325] = 37.402379371890226877894747303799;
    cOute[326] = 41.474250832760070295535115292296;
    cOute[327] = 50.428163236505760380623542005196;
    cOute[328] = 59.680552083361511961356882238761;
    cOute[329] = 59.852113921407159580212464788929;
    cOute[330] = 39.179032773860733129822619957849;
    cOute[331] = 41.811314587345911775173590285704;
    cOute[332] = 50.356538579204247696452512172982;
    cOute[333] = 58.774300532816020847803883953020;
    cOute[334] = 59.597627681225183948754420271143;
    cOute[335] = 39.406395497257157956028095213696;
    cOute[336] = 44.825102288614836254510009894148;
    cOute[337] = 52.657820630486305901740706758574;
    cOute[338] = 59.794778086404697603484237333760;
    cOute[339] = 60.100182642925126685895520495251;
    cOute[340] = 42.611339542509021782734635053203;
    cOute[341] = 44.391221408048345153929403750226;
    cOute[342] = 53.332009390351906574778695357963;
    cOute[343] = 60.853340645721864632378128590062;
    cOute[344] = 60.640613144207172524602356133983;
    cOute[345] = 42.118156896870438288260629633442;
    cOute[346] = 42.936173932273213438293169019744;
    cOute[347] = 51.454083271583961334272316889837;
    cOute[348] = 61.602068974814933710604236694053;
    cOute[349] = 61.635605439989227249952818965539;
    cOute[350] = 41.355446593145934741642122389749;
    cOute[351] = 45.154835186249350442722061416134;
    cOute[352] = 53.934573492916008774500369327143;
    cOute[353] = 61.479428737944225247247231891379;
    cOute[354] = 62.496466130853384868260036455467;
    cOute[355] = 42.296542227793530344115424668416;
    cOute[356] = 44.609827243049956280174228595570;
    cOute[357] = 55.735855544198066979788563912734;
    cOute[358] = 62.381816065803597837202687514946;
    cOute[359] = 64.297748182135435968120873440057;
    cOute[360] = 45.276824279075604806621413445100;
    cOute[361] = 46.919757744332009963272867025807;
    cOute[362] = 55.646604388352592707178700948134;
    cOute[363] = 61.820729082531180154091998701915;
    cOute[364] = 62.435289780667083903153979917988;
    cOute[365] = 44.405769894112616213988076196983;
    cOute[366] = 48.338018532196123544508736813441;
    cOute[367] = 57.370796778983979891108901938424;
    cOute[368] = 64.145884094059482549710082821548;
    cOute[369] = 64.639059454315955122183368075639;
    cOute[370] = 44.837305710204439890276262303814;
    cOute[371] = 45.757974879451388972029235446826;
    cOute[372] = 55.097968539976882595965435029939;
    cOute[373] = 66.011908559723991629653028212488;
    cOute[374] = 65.224718692169346923037664964795;
    cOute[375] = 50.278801168276764599340822314844;
    cOute[376] = 52.031515434590367874534422298893;
    cOute[377] = 46.242238396392870924955786904320;
    cOute[378] = 60.024368413161717228376801358536;
    cOute[379] = 65.091580333504197142246994189918;
    cOute[380] = 50.551074649294740481764165451750;
    cOute[381] = 49.378169786328349744053411995992;
    cOute[382] = 48.841099362782550485917454352602;
    cOute[383] = 64.396056515318008450776687823236;
    cOute[384] = 67.900795252813836100358457770199;
    cOute[385] = 51.283468462841263146856363164261;
    cOute[386] = 57.503453918035411618348007323220;
    cOute[387] = 58.920234028976558704471244709566;
    cOute[388] = 65.444512699459224336351326201111;
    cOute[389] = 68.402116259907103312798426486552;
    cOute[390] = 47.666368136767893304295284906402;
    cOute[391] = 49.627240152744711565446777967736;
    cOute[392] = 54.439016243187403176762018119916;
    cOute[393] = 62.547434335520357251425593858585;
    cOute[394] = 67.947202550405805254740698728710;
    cOute[395] = 48.467650188049951509583479491994;
    cOute[396] = 53.078479220331139742938830750063;
    cOute[397] = 58.068352554867665560323075624183;
    cOute[398] = 62.806368364710884577561955666170;
    cOute[399] = 63.547086016941200625751662300900;
    cOute[400] = 53.720260546499652321017492795363;
    cOute[401] = 54.219321017527640549360512522981;
    cOute[402] = 54.367005278187669148337590740994;
    cOute[403] = 62.194162745942996650683198822662;
    cOute[404] = 64.891575926698536136427719611675;
    cOute[405] = 54.124096667969418206212139921263;
    cOute[406] = 51.562456757081271518927678698674;
    cOute[407] = 52.027727915395040270141180371866;
    cOute[408] = 64.669784017275063092711206991225;
    cOute[409] = 67.850381221791039365598408039659;
    cOute[410] = 53.510038238517104502989241154864;
    cOute[411] = 51.833453641206553186293604085222;
    cOute[412] = 56.734362623133371528183488408104;
    cOute[413] = 65.971066068557121297999401576817;
    cOute[414] = 67.530500249065497087030962575227;
    cOute[415] = 51.672778393178212752445688238367;
    cOute[416] = 53.634735692488597180727083468810;
    cOute[417] = 56.690940562300845328991272253916;
    cOute[418] = 66.011496569839167136706237215549;
    cOute[419] = 67.252214222069497395750659052283;
    cOute[420] = 52.474060444460256746879167621955;
    cOute[421] = 56.262530630927123809215117944404;
    cOute[422] = 58.413784272494389426810812437907;
    cOute[423] = 64.312778621121225341994431801140;
    cOute[424] = 65.973035031377634140881127677858;
    cOute[425] = 53.275342495742314952167362207547;
    cOute[426] = 59.062347414258248079477198189124;
    cOute[427] = 60.255285602161841040924628032371;
    cOute[428] = 65.114060672403269336427911184728;
    cOute[429] = 66.774317082659678135314607061446;
    cOute[430] = 54.076624547024373157455556793138;
    cOute[431] = 55.854284339124284031186107313260;
    cOute[432] = 57.478948757474377373455354245380;
    cOute[433] = 66.415342723685327541716105770320;
    cOute[434] = 68.075599133941736340602801647037;
    cOute[435] = 54.877906598306417151889036176726;
    cOute[436] = 58.488163008562899847220251103863;
    cOute[437] = 62.483476284874861050866456935182;
    cOute[438] = 68.022491111133589924975240137428;
    cOute[439] = 69.876881185223780335036281030625;
    cOute[440] = 55.679188649588475357177230762318;
    cOute[441] = 57.442122961633920397162000881508;
    cOute[442] = 60.277780115851534503690345445648;
    cOute[443] = 66.357713230923195624200161546469;
    cOute[444] = 69.178163236505838540324475616217;
    cOute[445] = 56.480470700870533562465425347909;
    cOute[446] = 59.953313989378550274977897061035;
    cOute[447] = 61.986393017329284305105829844251;
    cOute[448] = 66.819188877531473735871259123087;
    cOute[449] = 67.979445287787882534757954999804;
    cOute[450] = 57.281752752152620189463050337508;
    cOute[451] = 62.232210728506935026871360605583;
    cOute[452] = 64.519542626534573059871036093682;
    cOute[453] = 68.120470928813531941159453708678;
    cOute[454] = 68.780727339069940740046149585396;
    cOute[455] = 58.083034803434692605605960125104;
    cOute[456] = 60.029415551538214401716686552390;
    cOute[457] = 62.319457119893364449580985819921;
    cOute[458] = 69.421752980095575935592933092266;
    cOute[459] = 71.082009390351998945334344170988;
    cOute[460] = 58.884316854716793443458300316706;
    cOute[461] = 60.859073264973140737765788799152;
    cOute[462] = 67.259319143215037684058188460767;
    cOute[463] = 70.487794118402888443597475998104;
    cOute[464] = 71.383291441634042939767823554575;
    cOute[465] = 59.685598905998823227037064498290;
    cOute[466] = 61.367151880698578736428316915408;
    cOute[467] = 64.637309453080661114654503762722;
    cOute[468] = 68.392431912235480240269680507481;
    cOute[469] = 70.184573492916101145056018140167;
    cOute[470] = 61.084395180598598074084293330088;
    cOute[471] = 62.786800027424852999047288903967;
    cOute[472] = 63.719091504362744160516740521416;
    cOute[473] = 68.658835388387402076659782323986;
    cOute[474] = 69.420023485956548370268137659878;
    cOute[475] = 62.102663008562878133034246275201;
    cOute[476] = 68.361561951491083277687721420079;
    cOute[477] = 70.160323313500938979814236517996;
    cOute[478] = 71.288356739701910669282369781286;
    cOute[479] = 72.075734375424076461058575659990;
    cOute[480] = 62.089445059844955210337502649054;
    cOute[481] = 64.344051671017012949960189871490;
    cOute[482] = 68.414487251875414131063735112548;
    cOute[483] = 73.043337926608188581667491234839;
    cOute[484] = 74.088419646762247339211171492934;
    cOute[485] = 62.890727111126977888488909229636;
    cOute[486] = 64.000983521383375318691832944751;
    cOute[487] = 68.406372299578947604459244757891;
    cOute[488] = 73.879340207451733135712856892496;
    cOute[489] = 74.797050688601714796277519781142;
    cOute[490] = 63.692009162409114253478037426248;
    cOute[491] = 67.079835346762379799656628165394;
    cOute[492] = 69.156426585769324333341501187533;
    cOute[493] = 73.336091461769839838780171703547;
    cOute[494] = 73.563890457909892006682639475912;
    cOute[495] = 65.202707135735707311141595710069;
    cOute[496] = 67.196722577770302109456679318100;
    cOute[497] = 69.361547734540906162692408543080;
    cOute[498] = 72.229873883998351402624393813312;
    cOute[499] = 72.297479284812169453289243392646;
    cOute[500] = 72.858000169805819723478634841740;
    cOute[501] = 73.308106218924990571395028382540;
    cOute[502] = 72.745654295818638956916402094066;
    cOute[503] = 79.257245215690062423163908533752;
    cOute[504] = 83.702500970027443827348179183900;
    cOute[505] = 69.794737950204194021353032439947;
    cOute[506] = 70.247120999146972053495119325817;
    cOute[507] = 69.878266871325365627853898331523;
    cOute[508] = 78.056341926560506294663355220109;
    cOute[509] = 86.703324610076450085216492880136;
    cOute[510] = 66.897137367537240493220451753587;
    cOute[511] = 68.283300614501555969582113903016;
    cOute[512] = 71.217650188050058090993843507022;
    cOute[513] = 83.853646789949678463926829863340;
    cOute[514] = 87.754251420436119701662391889840;
    cOute[515] = 67.698419418819284487653931137174;
    cOute[516] = 70.319774835904851784107449930161;
    cOute[517] = 71.518932239332116296282038092613;
    cOute[518] = 82.129788604922794092999538406730;
    cOute[519] = 87.747561892114106285589514300227;
    cOute[520] = 70.131380162090579233336029574275;
    cOute[521] = 71.991599325850188506592530757189;
    cOute[522] = 71.579025530505703045491827651858;
    cOute[523] = 81.838419646762247339211171492934;
    cOute[524] = 83.498676057018656138097867369652;
    cOute[525] = 79.871091390415415389725239947438;
    cOute[526] = 80.705687020721825319924391806126;
    cOute[527] = 81.199395303287715819351433310658;
    cOute[528] = 83.031068791234631021325185429305;
    cOute[529] = 84.283976210933630568433727603406;
    cOute[530] = 73.270482707493371776763524394482;
    cOute[531] = 74.057832409166238107900426257402;
    cOute[532] = 74.361232034436298476975935045630;
    cOute[533] = 81.640869799326381439641409087926;
    cOute[534] = 83.862091709582770704400900285691;
    cOute[535] = 70.903547623947488887097279075533;
    cOute[536] = 74.563804034203897685983974952251;
    cOute[537] = 75.724060444460320695725386030972;
    cOute[538] = 84.209543137505676213550032116473;
    cOute[539] = 85.402522210864816543107735924423;
    cOute[540] = 73.204829675229547092385473661125;
    cOute[541] = 74.315086085485944522588397376239;
    cOute[542] = 74.025342495742364690158865414560;
    cOute[543] = 84.257552017955504197743721306324;
    cOute[544] = 86.703804262146860537541215308011;
    cOute[545] = 75.758320368331567351560806855559;
    cOute[546] = 76.392051025494012606031901668757;
    cOute[547] = 76.988929455238277910211763810366;
    cOute[548] = 80.828532885020351272942207287997;
    cOute[549] = 84.572597550289074774809705559164;
    cOute[550] = 76.572160607809692578484828118235;
    cOute[551] = 77.406897555830809665167180355638;
    cOute[552] = 76.514495901316607273656700272113;
    cOute[553] = 84.146111954454553938376193400472;
    cOute[554] = 86.306368364710962737262889277190;
    cOute[555] = 76.184498453448171062518667895347;
    cOute[556] = 76.463507074318130207757349126041;
    cOute[557] = 76.247456859693471642458462156355;
    cOute[558] = 84.947394005736612143664387986064;
    cOute[559] = 88.745018869878038003662368282676;
    cOute[560] = 75.936137690593341176281683146954;
    cOute[561] = 78.137011844098481105902465060353;
    cOute[562] = 76.405584725021043368542450480163;
    cOute[563] = 87.248676057018656138097867369652;
    cOute[564] = 90.105125709617567508757929317653;
    cOute[565] = 77.914811629314385754696559160948;
    cOute[566] = 82.134499253277695629549270961434;
    cOute[567] = 79.315079033450970769081322941929;
    cOute[568] = 87.049958108300714343386061955243;
    cOute[569] = 89.565214518557112910457362886518;
    cOute[570] = 80.885441264041261888451117556542;
    cOute[571] = 82.971791690344943503987451549619;
    cOute[572] = 81.547801633450703207017795648426;
    cOute[573] = 87.283697573699114968803769443184;
    cOute[574] = 89.511496569839167136706237215549;
    cOute[575] = 80.448552906279076069040456786752;
    cOute[576] = 80.776940494358740352254244498909;
    cOute[577] = 80.782845206169781704375054687262;
    cOute[578] = 87.652522210864816543107735924423;
    cOute[579] = 88.312778621121225341994431801140;
    cOute[580] = 80.791542899080297956970753148198;
    cOute[581] = 81.094389212868861704919254407287;
    cOute[582] = 80.897256037492752511752769351006;
    cOute[583] = 88.715743310589772363528027199209;
    cOute[584] = 91.801498123238019388736574910581;
    cOute[585] = 81.170521611336127421054698061198;
    cOute[586] = 83.576430189842696449886716436595;
    cOute[587] = 81.555149167385778241623484063894;
    cOute[588] = 89.921963272353238494360994081944;
    cOute[589] = 94.133299019350474168277287390083;
    cOute[590] = 83.022178176798362869703851174563;
    cOute[591] = 85.303947293339163593373086769134;
    cOute[592] = 81.225314900556938368936243932694;
    cOute[593] = 90.056368364710962737262889277190;
    cOute[594] = 91.989978907824493603584414813668;
    cOute[595] = 84.220772898529332906036870554090;
    cOute[596] = 85.010948834513811789292958565056;
    cOute[597] = 84.231305181277434712683316320181;
    cOute[598] = 90.357650415993020942551083862782;
    cOute[599] = 91.017906826249429741437779739499;
    cOute[600] = 84.474421422108122214922332204878;
    cOute[601] = 86.070752815020739490137202665210;
    cOute[602] = 85.285400342818945773615268990397;
    cOute[603] = 89.515418981676361909194383770227;
    cOute[604] = 91.073522804863856094925722572953;
    cOute[605] = 83.365211246261409883118176367134;
    cOute[606] = 83.476973988195496190201083663851;
    cOute[607] = 84.188187218619631835281325038522;
    cOute[608] = 90.584144969834298422028950881213;
    cOute[609] = 93.027819919370941192937607411295;
    cOute[610] = 82.922778393178262490437191445380;
    cOute[611] = 83.583034803434671289323887322098;
    cOute[612] = 83.243291213691080088210583198816;
    cOute[613] = 90.207792998515529347969277296215;
    cOute[614] = 93.379474320738211190473521128297;
    cOute[615] = 86.308060444460295457247411832213;
    cOute[616] = 87.160866854716687157633714377880;
    cOute[617] = 86.264073264973120558352093212306;
    cOute[618] = 91.944116201968455470705521292984;
    cOute[619] = 94.077810805972944763198029249907;
    cOute[620] = 87.917202617174865508786751888692;
    cOute[621] = 88.077459027431274307673447765410;
    cOute[622] = 87.983528854854768042059731669724;
    cOute[623] = 93.232098430986837911405018530786;
    cOute[624] = 93.658075341243247180500475224108;
}