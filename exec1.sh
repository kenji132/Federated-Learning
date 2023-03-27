for method in fedavg trimmed_mean
do
    for alpha in 0.10 0.30 1e+20
    do
        python main.py --method $method --dirichlet_alpha $alpha
    done
done
