# %%
import sympy as sp
from IPython.display import display

# %%
k1, k2, t, t1, omega = sp.symbols("kappa_1 kappa_2 t t_1 Omega", real=True)
L0 = sp.Matrix(
    [[-k1 / 2, 0, 0, k1 / 2], [0, -k2, 0, 0], [0, 0, -k2, 0], [k1 / 2, 0, 0, -k1 / 2]]
)
L0
# %%
expL0 = sp.exp(L0 * t)
expL0
# %%
L1 = sp.Matrix([[0, 1, 1, 0], [-1, 0, 0, 1], [-1, 0, 0, 1], [0, -1, -1, 0]])
L1
# %%
sp.simplify(sp.exp(L1 * t))
# %%
to_integ = sp.simplify(sp.exp(L0 * (t - t1)) @ L1 @ sp.exp(L0 * t1))
to_integ
# %%
correction1 = sp.Integral(to_integ, (t1, 0, t)).doit()
correction1
# %%
I1 = []
for i in range(4):
    lst = []
    for j in range(4):
        term = correction1[i, j]
        if term == 0:
            lst.append(0)
        else:
            lst.append(term.args[0][0])
    I1.append(lst)
I1 = sp.Matrix(I1)
I1

# %%
t2 = sp.symbols("t_2", real=True, positive=True)
to_integ = sp.expand(sp.exp(L0 * (t - t2)) @ L1 @ I1.subs(t, t2))
to_integ

# %%
correction2 = sp.Integral(to_integ, (t2, 0, t)).doit()
correction2[0, 0]


# %%
################# With membrane interaction #################
k1, k2, t = sp.symbols("kappa_1 kappa_2 t", real=True, positive=True)
L0 = sp.Matrix(
    [[-k1 / 2, 0, 0, k1 / 2], [0, -k2, 0, 0], [0, 0, -k2, 0], [k1 / 2, 0, 0, -k1 / 2]]
)
L0
# %%
expL0 = sp.exp(L0 * t)
expL0
# %%
aopl, aopr, aopdl, aopdr = sp.symbols("a_l a_r a_l^\\dag a_r^\\dag", commutative=False)
Omega = sp.symbols("\\Omega", real=True, positive=True)
L1 = (
    Omega
    / 2
    * sp.Matrix(
        [
            [0, aopdr, aopl, 0],
            [-aopr, 0, 0, aopl],
            [-aopdl, 0, 0, aopdr],
            [0, -aopdl, -aopr, 0],
        ]
    )
)
L1
# %%
t1 = sp.symbols("t_1", real=True, positive=True)
to_integ = sp.expand(sp.exp(L0 * (t - t1)) @ L1 @ sp.exp(L0 * t1))
to_integ
# %%
correction1 = sp.Integral(to_integ, (t1, 0, t)).doit()
correction1
# %%
I1 = []
for i in range(4):
    lst = []
    for j in range(4):
        term = correction1[i, j]
        if term == 0:
            lst.append(0)
            continue
        val = 0
        for el in term.args:
            if el.is_Mul:
                val += el
            else:
                val += el.args[0][0]
        val = val.expand()
        lst.append(val)
    I1.append(lst)
I1 = sp.Matrix(I1)
I1
# %%
# tau_1 = (1 - sp.exp(-k1 * t)) / k1
# tau_2 = (1 - sp.exp(-k2 * t)) / k2
tau_1, tau_2 = sp.symbols("tau_1 tau_2", real=True, positive=True)
tau_sigma = sp.symbols("\\tau_{\\Sigma}", real=True, positive=True)
# tau_sigma = (k1 * tau_1 - k2 * tau_2) / (k1 - k2)
a_sigma = (aopr + aopl) / 2
a_delta = (aopr - aopl) / 2
ad_sigma = (aopdr + aopdl) / 2
ad_delta = (aopdr - aopdl) / 2
# a_sigma, ad_sigma, a_delta, ad_delta = sp.symbols(
#     "a_Sigma a^\\dagger_Sigma a_Delta a^\\dagger_Delta", commutative=False
# )
a_plus = tau_sigma * a_sigma + tau_2 * a_delta
a_minus = tau_sigma * a_sigma - tau_2 * a_delta
ad_plus = tau_sigma * ad_sigma + tau_2 * ad_delta
ad_minus = tau_sigma * ad_sigma - tau_2 * ad_delta

I1_ana = (
    Omega
    / 2
    * sp.Matrix(
        [
            [0, ad_plus, a_minus, 0],
            [-a_plus, 0, 0, a_minus],
            [-ad_minus, 0, 0, ad_plus],
            [0, -ad_minus, -a_plus, 0],
        ]
    )
)
display(I1_ana)
display((I1 - I1_ana).expand())

# %%
########################## 2nd order perturbation ##########################
t2 = sp.symbols("t_2", real=True, positive=True)
to_integ = sp.expand(sp.exp(L0 * (t - t2)) @ L1 @ I1.subs(t, t2))
correction2 = sp.Integral(to_integ, (t2, 0, t)).doit()
correction2
# %%
I2 = []
for i in range(4):
    lst = []
    for j in range(4):
        term = correction2[i, j]
        if term == 0:
            lst.append(0)
            continue
        val = 0
        for el in term.args:
            if el.is_Mul:
                val += el
            else:
                val += el.args[0][0]
        val = val.expand()
        lst.append(val)
    I2.append(lst)
I2 = sp.Matrix(I2)
I2

# %%
C12, C13, C21, C24, C31, C34, C42, C43 = sp.symbols(
    "C_{12} C_{13} C_{21} C_{24} C_{31} C_{34} C_{42} C_{43}", commutative=False
)
V11, V14, V22, V23, V32, V33, V41, V44 = sp.symbols(
    "V_{11} V_{14} V_{22} V_{23} V_{32} V_{33} V_{41} V_{44}", commutative=False
)
rhom = sp.symbols("\\rho_m", commutative=False)
I1_ana_2 = sp.Matrix(
    [[0, C12, C13, 0], [C21, 0, 0, C24], [C31, 0, 0, C34], [0, C42, C43, 0]]
)
I2_ana_2 = sp.Matrix(
    [[V11, 0, 0, V14], [0, V22, V23, 0], [0, V32, V33, 0], [V41, 0, 0, V44]]
)

pm = sp.Symbol("\\pm", real=True, positive=True)
mat_pm = sp.Matrix([[1, pm, pm, 1]]) / 2
p = sp.Symbol("p", real=True, positive=True)
rho_tot = sp.Matrix([(1 - p) * rhom, 0, 0, p * rhom])
evol = I1_ana
expr = ((mat_pm @ evol @ rho_tot)[0, 0] / rhom).expand()
display(expr)
# %%
factors_a, factors_ad = [aopl, aopdr], [aopdl, aopr]
vals_left, vals_right = 0, 0
for term in expr.args:
    v = 1
    rest = 1
    for el in term.args:
        if el in factors_a:
            v *= el
        elif el in factors_ad:
            v *= el
        else:
            rest *= el
    try:
        index = factors_a.index(v)
        vals_left += rest
    except ValueError:
        pass
    try:
        index2 = factors_ad.index(v)
        vals_right += rest
    except ValueError:
        pass
m1, m2 = vals_left.simplify(), vals_right.simplify()
display(m1, m2)
eta = sp.symbols("\\eta", real=True, positive=True)
display((m1 * (m1 + m2)).expand().subs({pm: 1, p: (1 + eta) / 2}).simplify())
display((m2 * (m1 + m2)).expand().subs({pm: 1, p: (1 + eta) / 2}).simplify())
display(vals_left.simplify().subs({pm: 1, tau_sigma: t, tau_2: t, p: 1}))
display(vals_right.simplify().subs({pm: 1, tau_sigma: t, tau_2: t, p: 1}))


# %%
evol = I2
expr = ((mat_pm @ evol @ rho_tot)[0, 0] / rhom).expand()
display(expr)
# %%
factors_down = [aopl * aopdr, aopdr * aopl, aopdl * aopl, aopr * aopdr]
factors_up = [aopdl * aopr, aopr * aopdl, aopl * aopdl, aopdr * aopr]
vals_down = [0, 0, 0, 0]
vals_up = [0, 0, 0, 0]
ops = [aopl, aopdr, aopdl, aopr]
for factors, vals in zip([factors_down, factors_up], [vals_down, vals_up]):
    for term in expr.args:
        v = 1
        rest = 1
        for el in term.args:
            if el in ops:
                v *= el
            else:
                rest *= el
        try:
            index = factors.index(v)
            vals[index] += rest
        except ValueError:
            pass
        try:
            index2 = (factors_down + factors_up).index(v)
        except ValueError:
            print(term, "not in list")
    display(vals[0] - vals[1])
    display(vals[0] + vals[1])
    display(vals[2])
    display(vals[3])
# %%
kappa_down_ana = (
    Omega**2
    / 4
    * (
        (2 * p - 1)
        / (k1 - k2)
        * ((1 - sp.exp(-k2 * t)) / k2 - (1 - sp.exp(-k1 * t)) / k1)
        + 1 / k2 * (t - (1 - sp.exp(-k2 * t)) / k2)
    )
)
display(kappa_down_ana)
display((kappa_down_ana.expand() - (vals_down[0] + vals_down[1])).simplify())
display(kappa_down_ana.limit(k1, 0).limit(k2, 0))
kappa_up_ana = (
    Omega**2
    / 4
    * (
        -(2 * p - 1)
        / (k1 - k2)
        * ((1 - sp.exp(-k2 * t)) / k2 - (1 - sp.exp(-k1 * t)) / k1)
        + 1 / k2 * (t - (1 - sp.exp(-k2 * t)) / k2)
    )
)
display(kappa_up_ana)
display((kappa_up_ana.expand() - (vals_up[0] + vals_up[1])).simplify())
display(kappa_up_ana.limit(k1, 0).limit(k2, 0))
# %%
ma, mad, da, dad = sp.symbols("M[a] M[a^{\\dag}] D[a] D[a^{\\dag}]", commutative=False)
mat, madt, dat, dadt = sp.symbols(
    "M[a(t)] M[a^{\\dag}(t)] D[a(t)] D[a^{\\dag}(t)]", commutative=False
)
m1, m2, l1, l2 = sp.symbols("m_1 m_2 l_1 l_2")
kplus = (1 + m1 * ma + m2 * mad + l1 * da + l2 * dad) / 2
kminus = (1 - m1 * ma - m2 * mad + l1 * da + l2 * dad) / 2
kplust = (1 + m1 * mat + m2 * madt + l1 * dat + l2 * dadt) / 2
kminust = (1 - m1 * mat - m2 * madt + l1 * dat + l2 * dadt) / 2
correl = (
    kplust * kplus + kminust * kminus - kplust * kminus - kminust * kplus
).expand()
display(correl)
# %%
cada, caad = sp.symbols(r"Re(C_{{a^\dagger}a}) Re(C_{aa^\dagger})")
correl.subs(
    {mat * ma: cada, madt * ma: cada, mat * mad: caad, madt * mad: caad}
).expand()
# %%

theta = sp.symbols("\\theta", real=True, positive=True)
correl.subs(
    {mat * ma: cada, madt * ma: cada, mat * mad: caad, madt * mad: caad}
).expand().subs({m2: 0, m1: theta / 2})
# %%
