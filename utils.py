import numpy as np
from desc.vmec import VMECIO


class NEOWrapper:
    """Class to easily make NEO and BOOZxform inputs from DESC equilibria."""

    def __init__(self, basename, eq=None, ns=None, M_booz=None, N_booz=None):

        self.basename = basename
        self.M_booz = M_booz
        self.N_booz = N_booz

        if eq:
            self.build(eq, basename, ns=ns)

    def build(self, eq, basename, **kwargs):
        """
        Pass as input an already-solved Equilibrium from DESC
        """
        # equilibrium parameters
        self.eq = eq
        self.sym = eq.sym
        self.L = eq.L
        self.M = eq.M
        self.N = eq.N
        self.NFP = eq.NFP
        self.spectral_indexing = eq.spectral_indexing
        self.pressure = eq.pressure
        self.iota = eq.iota
        self.current = eq.current

        # wout parameters
        self.ns = kwargs.get("ns", 256)

        # booz parameters
        if self.M_booz is None:
            self.M_booz = 3 * eq.M + 1
        if self.N_booz is None:
            self.N_booz = 3 * eq.N

        # basename for files
        self.basename = basename

    def save_VMEC(self):

        self.eq.solved = True  # must set this for NEO to run correctly

        print(f"Saving VMEC wout file to wout_{self.basename}.nc")
        VMECIO.save(self.eq, f"wout_{self.basename}.nc", surfs=self.ns, verbose=0)

    def write_booz(self):
        """Write BOOZ_XFORM input file."""
        print(f"Writing BOOZ_XFORM input file to in_booz.{self.basename}")
        with open(f"in_booz.{self.basename}", "w+") as f:
            f.write("{} {}\n".format(self.M_booz, self.N_booz))
            f.write(f"'{self.basename}'\n")
            f.write(
                "\n".join([str(x) for x in range(2, self.ns + 1)])
            )  # surface indices

    def write_neo(self, N_particles=150):
        """Write NEO input file."""
        print(f"Writing NEO input file neo_in.{self.basename}")
        fname = f"neo_in.{self.basename}"
        with open(fname, "w+") as f:
            f.write("'#'\n'#'\n'#'\n")
            f.write(f" boozmn_{self.basename}.nc\n")  # booz out file
            f.write(f" neo_out.{self.basename}\n")  # desired NEO out file
            f.write(f" {self.ns-1}\n")  # number of surfaces
            f.write(
                " ".join([str(x) for x in range(2, self.ns + 1)]) + "\n"
            )  # surface indices
            f.write(
                " 300 ! number of theta points\n 300 ! number of zeta points"
                + f"\n 0\n 0\n {N_particles} ! number of test particles\n 1 ! 1 = singly trapped particles\n"
                + " 0.001 ! integration accuracy\n 100 ! number of poloidal bins\n 10 ! integration steps per field period"
                + "\n 500 ! min number of field periods\n 5000 ! max number of field periods\n"
            )  # default values
            f.write(
                " 0\n 1\n 0\n 0\n 2 ! 2 = reference |B| used is max on each surface \n 0\n 0\n 0\n 0\n 0\n"
            )
            f.write("'#'\n'#'\n'#'\n")
            f.write(" 0\n")
            f.write(f"neo_cur_{self.basename}\n")
            f.write(" 200\n 2\n 0\n")
        return fname


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def read_neo_out(fname):
    # import all data from text file as an array
    with open(f"{fname}") as f:
        array = np.array([[float(x) for x in line.split()] for line in f])

    eps_eff = array[:, 1]  # epsilon_eff^(3/2) is the second column
    nans, x = nan_helper(eps_eff)  # find NaN values

    # replace NaN values with linear interpolation
    eps_eff[nans] = np.interp(x(nans), x(~nans), eps_eff[~nans])

    return eps_eff


"""example usage
from utils import NEOWrapper

wrap_surf = NEOWrapper(basename=f"QI_fixed_surf_r{r}",eq=eq_surf,ns=ns)
wrap_nae = NEOWrapper(basename=f"QI_nae_r{r}",eq=desc_eq,ns=ns)

for wrapper in [wrap_surf, wrap_nae]:
    wrapper.write_booz()
    wrapper.write_neo()
    wrapper.save_VMEC()
    
"""
