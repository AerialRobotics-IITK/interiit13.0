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
#include "acados_sim_solver_broken.h"

#define NX     BROKEN_NX
#define NZ     BROKEN_NZ
#define NU     BROKEN_NU
#define NP     BROKEN_NP


int main()
{
    int status = 0;
    broken_sim_solver_capsule *capsule = broken_acados_sim_solver_create_capsule();
    status = broken_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = broken_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = broken_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = broken_acados_get_sim_out(capsule);
    void *acados_sim_dims = broken_acados_get_sim_dims(capsule);

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

  
    x_current[0] = -0.4844725728034973;
    x_current[1] = 0.02362531796097755;
    x_current[2] = -5.104054115712643;
    x_current[3] = 0.9999342560768129;
    x_current[4] = -0.0017752539133653045;
    x_current[5] = 0.0009387252503074706;
    x_current[6] = -0.011288703419268131;
    x_current[7] = -0.02380962483584881;
    x_current[8] = 0.022638602182269096;
    x_current[9] = 0.0394425205886364;
    x_current[10] = -0.00015435315435752273;
    x_current[11] = 0.00016778570716269314;
    x_current[12] = -0.00014230978558771312;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    // set parameters
    double p[NP];
    p[0] = -0.4844725728034973;
    p[1] = 0.02362531796097755;
    p[2] = -5.104054115712643;
    p[3] = 0.9999342560768129;
    p[4] = -0.0017752539133653045;
    p[5] = 0.0009387252503074706;
    p[6] = -0.011288703419268131;
    p[7] = -0.02380962483584881;
    p[8] = 0.022638602182269096;
    p[9] = 0.0394425205886364;
    p[10] = -0.00015435315435752273;
    p[11] = 0.00016778570716269314;
    p[12] = -0.00014230978558771312;
    p[13] = 0;
    p[14] = 0;
    p[15] = 0;
    p[16] = 0;

    broken_acados_sim_update_params(capsule, p, NP);
  

  


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
        status = broken_acados_sim_solve(capsule);
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
    status = broken_acados_sim_free(capsule);
    if (status) {
        printf("broken_acados_sim_free() returned status %d. \n", status);
    }

    broken_acados_sim_solver_free_capsule(capsule);

    return status;
}
