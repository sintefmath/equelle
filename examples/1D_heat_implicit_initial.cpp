#include <iostream>
#include <cmath>
#include <utility>
#include <cstring>

// Solving the n x n system a x = b, upper triangular matrix ends up in uppertri,
// b_scratch must have length n, solution ends up in x, returns true if non-singular,
// false otherwise. Matrices in row-major order. 
bool GaussElimIfNonSingular(const int n,
			    const double * const a, double * const uppertri,
			    const double * const b, double * const b_scratch,
			    double * const x)
{
    memcpy(uppertri, a, n*n*sizeof(double));
    memcpy(b_scratch, b, n*sizeof(double));
    memset(x, 0, n*sizeof(double));
    for (int j=0; j<n; j++) {
        int maxrow=j;
        for (int i=j+1; i<n; i++)
            if (fabs(uppertri[i*n+j]) > fabs(uppertri[maxrow*n+j]))
                maxrow = i;
        for (int k=j; k<n; k++)
            std::swap(uppertri[j*n+k], uppertri[maxrow*n+k]);
        std::swap(b_scratch[j], b_scratch[maxrow]);
        if (fabs(uppertri[j*n+j]) < 1e-12) // Singular?
            return false;
        for (int i=j+1; i<n; i++) {
            const double tmp = uppertri[i*n+j] / uppertri[j*n+j];
            b_scratch[i] -= b_scratch[j] * tmp;
            for (int k=n-1; k>=j; k--)
                uppertri[i*n+k] -= uppertri[j*n+k] * tmp;
        }
    }
    x[n-1] = b_scratch[n-1] / uppertri[(n-1)*n+(n-1)];
    for (int i=n-2; i>=0; i--) {
        double tmp = uppertri[i*n+i+1] * x[i+1];
        for (int k=i+2; k<n; k++)
            tmp += uppertri[i*n+k] * x[k];
        x[i] = (b_scratch[i] - tmp) / uppertri[i*n+i];
    }
    return true;
}


double u[6]; // Length from "grid.n = 6", type and dimension from "grid.type = RegularCartesian1D"
double A[6][6]; // From "simulator.type = Implicit", "grid.n = 6" and "simulator.basis = Constant"

void solution_init(void)
{
    for (int i=0; i<6; i++)
        u[i] = 0.0; // From "initial.value = 0"
    u[0] = 0.5; // From "boundary.val0 = 0.5"
    u[5] = 1.5; // From "boundary.val1 = 1.5"
}

void system_init(const double s)
{
    const int stencil_length = 3;
    const double stencil[stencil_length] = {-s, 1.0+2.0*s, -s}; // From "simulator.stencil = (-s, 1+2s, -s)"
    for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++)
            A[i][j] = 0.0;
        // A[i][i] = 1.0; // got this in the stencil already
        if ( (i>0) && (i<5) )
            for (int j=0; j<3; j++)
                A[i][i-stencil_length/2+j] += stencil[j];
    }
    A[0][0] = 1.0; // From the "boundary.*" entries:
    A[5][5] = 1.0;
}

void init(void)
{
    // const double dx = 1.0/6.0; // From "dx = grid.getSpacing()"
    const double dx = 0.5; // Overriding with the number from the iPython notebook, so that we can compare the results!
    // const double dt = dx*dx/(2*0.3); // From "dt = MaxStableTimeStep"
    // Hmm... This was the max dt for the *explicit* stencil...
    const double dt = 5.0; // Trying a larger value (Note: Use the same as for the iPython notebook, so that we can compare results...)
    system_init( 0.3 * dt / (dx*dx) ); // From "physics.diffusion_constant = 0.3" and "s = physics.diffusion_constant * dt / (dx*dx)"
    solution_init();
}

void performOneTimeStep(void)
{
    double v[6], Ascratch[36], uScratch[6];

    const bool nonSingular = GaussElimIfNonSingular(6, (double *)A, Ascratch, u, uScratch, v);
    for (int i=0; i<6; i++)
        u[i] = v[i];
}


int main(int argc, char *argv[])
{
    init();
    for (int i=0; i<20; i++) {
        std::cout << i << ": \t";
        for (int j=0; j<6; j++)
             std::cout << u[j] << " ";
        std::cout << std::endl;
        performOneTimeStep();
    }
    return 0;
}

