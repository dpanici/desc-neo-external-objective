from desc.objectives import (
    ExternalObjective,
    ObjectiveFunction,
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixCurrent,
    FixPsi,
    ForceBalance,
)
from desc.examples import get
from desc.grid import LinearGrid
from utils import NEOWrapper, read_neo_out
import numpy as np
import subprocess
from desc.vmec_utils import make_boozmn_output


def neofun(eq, ns_save=25, ns_opt_inds=None):
    if ns_opt_inds is None:
        # NEO returns eps eff on all but axis surface
        ns_opt_inds = np.arange(ns_save - 1)
    wrapper = NEOWrapper(basename=f"temp", eq=eq, ns=ns_save)
    wrapper.write_booz()
    neo_input_name = wrapper.write_neo()
    print(neo_input_name)
    wrapper.save_VMEC()
    print("making boozmn in DESC")
    make_boozmn_output(
        eq,
        "boozmn_temp.nc",
        surfs=ns_save,
        M_booz=18,
        N_booz=12,
    )
    # run neo
    print("Running NEO")
    process = subprocess.run(
        ["module load stellopt/intel-2021.1/intel-mpi/2.0.0 && xneo temp"],
        shell=True,
        check=True,
        timeout=60,
    )
    if process.returncode > 0:
        return 1e3 * np.ones_like(ns_opt_inds)
    else:  # succeeded
        eps_eff = read_neo_out("neo_out.temp")
        out = eps_eff[ns_opt_inds]
        print(out)
        print(eps_eff)
        return np.atleast_1d(out)


eq = get("ESTELL")
ns = 25
obj = ObjectiveFunction(
    (
        ExternalObjective(
            eq,
            fun=neofun,
            dim_f=1,
            fun_kwargs={"ns_save": ns, "ns_opt_inds": np.array([13])},
            abs_step=1e-3,
        ),
    )
)

modes_fix_R = np.delete(
    eq.surface.R_basis.modes, eq.surface.R_basis.get_idx(M=1, N=1), axis=0
)
modes_fix_Z = np.delete(
    eq.surface.Z_basis.modes, eq.surface.Z_basis.get_idx(M=1, N=-1), axis=0
)
print(modes_fix_R)
print(modes_fix_R.shape)


constraints = (
    ForceBalance(eq),
    FixBoundaryR(eq, modes=modes_fix_R),
    FixBoundaryZ(eq, modes=modes_fix_Z),
    FixPressure(eq),
    FixCurrent(eq),
    FixPsi(eq),
)

eq_opt, _ = eq.optimize(
    objective=obj, constraints=constraints, maxiter=4, verbose=3, copy=True
)

eq_opt.save("ESTELL_neo_eps_eff_opt.h5")
