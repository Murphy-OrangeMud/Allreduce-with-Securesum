import numpy as np
import matplotlib.pyplot as plt

x = np.array([i for i in range(10)])

DATASET = "MNIST"
y_baseline_time_3_nodes = np.array([
    55.496260181,
    55.504357499,
    57.202688164,
    52.730917228,
    56.528380031,
    56.556346752,
    53.639776375,
    50.943821365,
    57.995617834,
    53.17509746
])

y_baseline_time_5_nodes = np.array([
    89.255163188,
    86.596311455,
    87.024234152,
    87.389202235,
    88.414468337,
    87.699294463,
    88.338768857,
    86.678176284,
    84.430441494,
    91.002582328
])

y_mpc_time_3_nodes = np.array([
    75.328155028,
    75.479092724,
    75.031881588,
    75.710582703,
    72.236946345,
    73.113141442,
    71.062052432,
    72.18308759,
    73.103529736,
    74.422542086
])

y_mpc_time_5_nodes = np.array([
    104.245060069,
    100.061482345,
    102.167957097,
    102.705596082,
    102.09344019,
    99.546697471,
    103.114154456,
    103.27413471,
    104.16604012,
    101.445775182
])

y_baseline_loss_3_nodes = np.array([
    0.5346739530813269,
    0.2733993945483204,
    0.20173343107010583,
    0.15449342754058368,
    0.13191907272056191,
    0.10953297375055104,
    0.09894512692442949,
    0.08813035293956692,
    0.0788918659330172,
    0.07228908467836741
])

y_baseline_loss_5_nodes = np.array([
    0.5391133964682618,
    0.2618299634700330,
    0.19774167846577864,
    0.15548275438680623,
    0.12663418940501286,
    0.10305307523425047,
    0.09315814557339763,
    0.08213755860924721,
    0.07989658704367078,
    0.07164495999847229
])

y_mpc_loss_3_nodes = np.array([
    0.5369474077968227,
    0.2617766002361005,
    0.2005506389241361,
    0.15343359292756212,
    0.13078253418090022,
    0.10770520080560408,
    0.0951328652547731,
    0.08415736165336983,
    0.07933248502087137,
    0.07469964802970004
])

y_mpc_loss_5_nodes = np.array([
    0.5393253702980777,
    0.26040512310961883,
    0.19839798198857655,
    0.15436749414463216,
    0.12687950311810708,
    0.10279586330289021,
    0.09321929536527022,
    0.08236976201102758,
    0.0784864403239529,
    0.07173947324166269
])

print(f"{DATASET} dataset time cost mean:\nBASELINE:\n \
    3 nodes: {np.mean(y_baseline_time_3_nodes)}\n  \
    5 nodes: {np.mean(y_baseline_time_5_nodes)}\n \
    SECURE SUM:\n 3 nodes: {np.mean(y_mpc_time_3_nodes)}\n \
    5 nodes: {np.mean(y_mpc_time_5_nodes)}\n")

DATASET = "CIFAR-10"
y_baseline_loss_3_nodes = np.array([
    2.2916073054450585,
    2.138588475640535,
    2.0142565653366167,
    1.929723437547083,
    1.871739893175793,
    1.8229834859257081,
    1.78124677894698,
    1.7327545571987815,
    1.6978048806827075,
    1.662503452985653
])

y_baseline_time_3_nodes = np.array([
    57.036549492,
    57.639296974,
    53.95564054,
    56.011131418,
    54.178682554,
    59.563117476,
    56.489322724,
    58.437492865,
    51.88403168,
    59.536257307
])

y_mpc_loss_3_nodes = np.array([
    2.2916070844424463,
    2.1385856191217147,
    2.014252603804735,
    1.9297262406169018,
    1.8717555732510853,
    1.8230022711477591,
    1.781265965937367,
    1.7327577539295034,
    1.697817794922317,
    1.6624896685482573
])

y_mpc_time_3_nodes = np.array([
    58.868564865,
    54.14448199,
    61.225979433,
    57.068079237,
    59.91889292,
    62.744664854,
    60.720962501,
    63.919963603,
    62.377994839,
    59.725266924
])

y_baseline_loss_5_nodes = np.array([
    2.2922344827651977,
    2.135065776705742,
    2.014990336000919,
    1.92774955034256,
    1.8681077536940576,
    1.8249654018878936,
    1.7800028538703918,
    1.7396804302930833,
    1.698289145231247,
    1.6614019787311554
])

y_baseline_time_5_nodes = np.array([
    92.033808213,
    90.680808833,
    93.094813207,
    93.360277911,
    96.844377236,
    93.328441528,
    97.615277312,
    101.093585029,
    96.670536041,
    95.932765116
])

y_mpc_loss_5_nodes = np.array([
    2.2922346872091293,
    2.1350659361481665,
    2.0149876537919043,
    1.9277455961704255,
    1.8680942469835282,
    1.8249580466747284,
    1.780000637471676,
    1.7396586692333222,
    1.6982638165354729,
    1.661409207880497
])

y_mpc_time_5_nodes = np.array([
    103.594144755,
    96.23282463,
    94.980955368,
    99.155539946,
    95.321762429,
    97.399277561,
    97.825522929,
    96.394730852,
    97.945277992,
    97.614999875
])

print(f"{DATASET} dataset time cost mean:\nBASELINE:\n \
    3 nodes: {np.mean(y_baseline_time_3_nodes)}\n  \
    5 nodes: {np.mean(y_baseline_time_5_nodes)}\n \
    SECURE SUM:\n 3 nodes: {np.mean(y_mpc_time_3_nodes)}\n \
    5 nodes: {np.mean(y_mpc_time_5_nodes)}\n")

DATASET = "CIFAR-100"
y_baseline_loss_3_nodes = np.array([
    4.605568784910726,
    4.602245586645092,
    4.583015900715172,
    4.529723886879025,
    4.3663234008049185,
    4.2221364770788385,
    4.152543170025727,
    4.0902577725705935,
    4.0264604716216885,
    3.9689410183231537
])

y_baseline_time_3_nodes = np.array([
    55.138928091,
    60.340616775,
    60.803610671,
    60.896149883,
    62.191687719,
    61.547845046,
    62.074539625,
    63.860451791,
    59.748365354,
    57.95321268
])

y_mpc_loss_3_nodes = np.array([
    4.605568753682095,
    4.602245438909651,
    4.583015393850485,
    4.529727957410836,
    4.366345968895055,
    4.222137439160863,
    4.152536328553553,
    4.0901994224759735,
    4.0264354544862995,
    3.9688723952103624
])

y_mpc_time_3_nodes = np.array([
    67.853858377,
    64.187837241,
    60.843856862,
    59.368188759,
    58.083658342,
    60.909065721,
    57.71708462,
    59.516320518,
    64.469896942,
    65.381132499
])

y_baseline_loss_5_nodes = np.array([
    4.605705865621567,
    4.602261787652969,
    4.582400766611099,
    4.5270809841156,
    4.359410085082054,
    4.2154809683561325,
    4.155205041766167,
    4.088082321882248,
    4.022507571578026,
    3.970545683503151
])

y_baseline_time_5_nodes = np.array([
    90.794749955,
    97.279935698,
    94.339449383,
    98.041362339,
    99.467030541,
    97.632635229,
    98.647757775,
    101.07043687,
    102.017976256,
    100.408216197
])

y_mpc_loss_5_nodes = np.array([
    4.605705871582031,
    4.6022619318962095,
    4.582401951551438,
    4.5270800530910495,
    4.359411848783493,
    4.21548229932785,
    4.1552240645885465,
    4.0880831772089,
    4.022549090385437,
    3.970600289106369
])

y_mpc_time_5_nodes = np.array([
    102.555760321,
    94.982093348,
    97.084623265,
    96.119051892,
    94.741618443,
    100.483405658,
    100.812696863,
    101.193361283,
    102.255759407,
    103.202800002
])

print(f"{DATASET} dataset time cost mean:\nBASELINE:\n \
    3 nodes: {np.mean(y_baseline_time_3_nodes)}\n  \
    5 nodes: {np.mean(y_baseline_time_5_nodes)}\n \
    SECURE SUM:\n 3 nodes: {np.mean(y_mpc_time_3_nodes)}\n \
    5 nodes: {np.mean(y_mpc_time_5_nodes)}\n")


# Loss: 3 nodes and 5 nodes in two graphs
# Time: 3 nodes and 5 nodes in two graphs
plt.figure()
plt.title("Loss in %s dataset of 3 nodes training" % DATASET)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(x, y_baseline_loss_3_nodes, label='baseline', marker='o')
plt.plot(x, y_mpc_loss_3_nodes, label='secure sum', marker='o')
plt.legend()
plt.savefig("./Loss_3_%s" % DATASET)

plt.figure()
plt.title("Loss in %s dataset of 5 nodes training" % DATASET)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(x, y_baseline_loss_5_nodes, label='baseline', marker='o')
plt.plot(x, y_mpc_loss_5_nodes, label='secure sum', marker='o')
plt.legend()
plt.savefig("./Loss_5_%s" % DATASET)

plt.figure()
plt.title("Time cost in %s dataset of 3 nodes training" % DATASET)
plt.xlabel("Epoch")
plt.ylabel("Time(s)")
plt.plot(x, y_baseline_time_3_nodes, label='baseline', marker='o')
plt.plot(x, y_mpc_time_3_nodes, label='secure sum', marker='o')
plt.legend()
plt.savefig("./Time_3_%s" % DATASET)

plt.figure()
plt.title("Time cost in %s dataset of 5 nodes training" % DATASET)
plt.xlabel("Epoch")
plt.ylabel("Time(s)")
plt.plot(x, y_baseline_time_5_nodes, label='baseline', marker='o')
plt.plot(x, y_mpc_time_5_nodes, label='secure sum', marker='o')
plt.legend()
plt.savefig("./Time_5_%s" % DATASET)

