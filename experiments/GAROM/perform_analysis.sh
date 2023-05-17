# SIMULATIONS ANALYSIS RESULTS
#
# 1. VIOLIN PLOT + UQ + Model Performance
#
# 2. Model convergence average
#
#
# ================================

# VIOLIN PLOT + UQ + Model Performance
# for seed in {300,};
# do
#     for reg in {1,0};
#     do

#         echo "LidCavity"
#         echo " "

#         python3 lid.py --encoding_dim=16 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/lid --print=True --regular=$reg
#         python3 lid.py --encoding_dim=64 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/lid --print=True --regular=$reg
#         python3 lid.py --encoding_dim=120 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/lid --print=True --regular=$reg

#         echo "Graetz simulation"
#         echo " "

#         python3 graetz.py --encoding_dim=16 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/graetz --print=True --regular=$reg
#         python3 graetz.py --encoding_dim=64 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/graetz --print=True --regular=$reg
#         python3 graetz.py --encoding_dim=120 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/graetz --print=True  --regular=$reg


#         echo "Gaussian"
#         echo " "

#         python3 gaussian.py --encoding_dim=4 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/gaussian --print=True --regular=$reg
#         python3 gaussian.py --encoding_dim=16 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/gaussian --print=True --regular=$reg
#         python3 gaussian.py --encoding_dim=64 --noise_dim=12 --seed=$seed --path=sim_${seed}/results_${reg}/gaussian --print=True --regular=$reg

#     done
# done

python analysis.py
# Model convergence average

