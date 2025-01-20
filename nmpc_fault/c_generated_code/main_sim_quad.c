/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_quad.h"

#define NX     QUAD_NX
#define NZ     QUAD_NZ
#define NU     QUAD_NU
#define NP     QUAD_NP


int main()
{
    int status = 0;
    quad_sim_solver_capsule *capsule = quad_acados_sim_solver_create_capsule();
    status = quad_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = quad_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = quad_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = quad_acados_get_sim_out(capsule);
    void *acados_sim_dims = quad_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;
    x_current[6] = 0.0;
    x_current[7] = 0.0;
    x_current[8] = 0.0;
    x_current[9] = 0.0;
    x_current[10] = 0.0;
    x_current[11] = 0.0;
    x_current[12] = 0.0;

  
    x_current[0] = -0.17957964539527893;
    x_current[1] = 0.10158510506153108;
    x_current[2] = -0.017356060445308685;
    x_current[3] = 0.9999977350234984;
    x_current[4] = 0.0005832163151353598;
    x_current[5] = 0.0011248650262132287;
    x_current[6] = -0.0017329048132523894;
    x_current[7] = -0.012883702293038368;
    x_current[8] = 0.0027503781020641327;
    x_current[9] = 0.030654096975922585;
    x_current[10] = 0.000006272348400671035;
    x_current[11] = 0.0002288236137246713;
    x_current[12] = -0.000027899319320567884;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    // set parameters
    double p[NP];
    p[0] = -0.17957964539527893;
    p[1] = 0.10158510506153108;
    p[2] = -0.017356060445308685;
    p[3] = 0.9999977350234984;
    p[4] = 0.0005832163151353598;
    p[5] = 0.0011248650262132287;
    p[6] = -0.0017329048132523894;
    p[7] = -0.012883702293038368;
    p[8] = 0.0027503781020641327;
    p[9] = 0.030654096975922585;
    p[10] = 0.000006272348400671035;
    p[11] = 0.0002288236137246713;
    p[12] = -0.000027899319320567884;
    p[13] = 0;
    p[14] = 0;
    p[15] = 0;
    p[16] = 0;

    quad_acados_sim_update_params(capsule, p, NP);
  

  


    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "u", u0);

        // solve
        status = quad_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);

    

        // print solution
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = quad_acados_sim_free(capsule);
    if (status) {
        printf("quad_acados_sim_free() returned status %d. \n", status);
    }

    quad_acados_sim_solver_free_capsule(capsule);

    return status;
}
