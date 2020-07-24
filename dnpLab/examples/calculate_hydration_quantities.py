import dnpLab
from dnpLab import create_workspace
import numpy as np

ws = create_workspace()

# Create test data.
hydration = {
    'T1':np.array([2.020153734009,
                            2.276836030132750,
                            2.3708172489377400,
                            2.4428968088189100,
                            2.5709096032675700]),
    'T1_power':np.array([0.000589495934876689,
                                  0.024242327290569100,
                                  0.054429505156431400,
                                  0.0862844940360515,
                                  0.11617812912435900]),
    'E':np.array([0.57794113752189,
                           -0.4688718613022250,
                           -0.5464528159680670,
                           -1.0725090541762200,
                           -1.4141203961920700,
                           -1.695789643686440,
                           -1.771840068080760,
                           -1.8420812985152700,
                           -1.97571340381877,
                           -2.091405209753480,
                           -2.1860546327712800,
                           -2.280712535872610,
                           -2.4709892163826400,
                           -2.5184316153191200,
                           -2.556110148443770,
                           -2.576413132701720,
                           -2.675593912859120,
                           -2.8153300703866400,
                           -2.897475156648710,
                           -3.0042154567120800,
                           -3.087886507216510]),
    'E_power':np.array([0.0006454923080882520,
                                 0.004277023425898170,
                                 0.004719543572446050,
                                 0.00909714298712173,
                                 0.01344187403986090,
                                 0.01896059941058610,
                                 0.02101937603827090,
                                 0.022335737104727900,
                                 0.026029715703921800,
                                 0.02917012237740640,
                                 0.0338523245243911,
                                 0.03820738749745440,
                                 0.04733370907740660,
                                 0.05269608016472140,
                                 0.053790874615060400,
                                 0.05697639350179900,
                                 0.06435487925718170,
                                 0.07909179437004270,
                                 0.08958910066880800,
                                 0.1051813598911370,
                                 0.11617812912435900])
}

# Parameters and configuratinos
hydration.update({
    'T10': 1.5,
    'T100': 2.0,
    'spin_C': 125,
    'field': 348.5,
    'smax_model': 'tethered',
    't1_interp_method': 'second_order'
})

# Add hydration to workspace
ws.add('hydration', hydration)

# Run hydration calculation
res = dnpLab.hydration.hydration(ws)

# Observe results, k_sigma, etc
print(''.join([f'{k} = {res[k]}\n' for k in ['k_sigma', 'krho', 'klow', 'tcorr']]))

assert abs(res['k_sigma'] - 20.18) < 0.01