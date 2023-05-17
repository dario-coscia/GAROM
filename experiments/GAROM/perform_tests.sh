# SIMULATIONS

for seed in {100,200,300,400,500};
do
    mkdir -p sim_${seed}

    for reg in {1,0};
    do

        mkdir -p sim_${seed}/results_${reg}

        echo "LidCavity"
        echo " "

        python3 lid.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=16 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed
        python3 lid.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=64 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed
        python3 lid.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=120 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed

        mkdir -p sim_${seed}/results_${reg}/lid
        mv discriminator_* generator_* lid_*.csv sim_${seed}/results_${reg}/lid

        echo "Graetz simulation"
        echo " "

        python3 graetz.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=16 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed
        python3 graetz.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=64 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed
        python3 graetz.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=120 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed

        mkdir -p sim_${seed}/results_${reg}/graetz
        mv discriminator_* generator_* graetz_*csv sim_${seed}/results_${reg}/graetz

        echo "Gaussian"
        echo " "

        python3 gaussian.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=4 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed
        python3 gaussian.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=16 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed
        python3 gaussian.py --epochs=20000 --print_every=1000 --print=True --encoding_dim=64 --noise_dim=12 --gamma=0.3 --lambda_k=0.001 --regular=$reg --seed=$seed

        mkdir -p sim_${seed}/results_${reg}/gaussian
        mv discriminator_* generator_* gaussian_*.csv sim_${seed}/results_${reg}/gaussian

    done
done