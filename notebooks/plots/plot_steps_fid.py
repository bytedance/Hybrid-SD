# %%
import matplotlib.pyplot as plt
import numpy as np
# Sample data
# sd1.4, bk-sdm-small,bk-sdm-tiny, ours-tiny, hybrid_ours(k=5), hybrid_ours(k=10) 
flops = [33.89, 21.78, 20.51, 15.14, 18.89, 22.64]  # Latency values
fid = [12.22,17.05 , 17.05, 16.71, 15.39, 13.75]  # Accuracy values
parameters = [859.52, 482.34, 323.38, 224.49, 224.49, 224.49]  # Number of parameters
#clip = [0.2993, , 0.2673, 0.2611, 0.2734, 0.2887]
#colors = ['#C82423', '#007bbb', '#2878B5', 'yellow', '#FFBE7A', '#FA9F8F'] 
#colors = ['#2792c3', '#007bbb', '#2878B5', '#aacf53', '#a8c97f', '#839b5c']
#colors = ['#2792c3', '#007bbb', '#2878B5', '#a2d7dd', '#c1e4e9', '#a0d8ef']  
colors = ['darkblue', '#007bbb', '#2878B5', '#867ba9', '#a59aca', '#674196'] 


size_scale = 10
# lcm data
# sd1.4, tiny, hybrid
flops_lcm = [10.86, 4.85, 7.86]
fid_lcm = [16.30, 23.42, 16.19]
parameters_lcm = [859.60, 224.6, 224.6]
colors_lcm = ['#d0576b', '#e83929', '#e60033']
colors_lcm = ['#b7282e', '#e83929', '#e60033']


flops = flops + flops_lcm
fid = fid + fid_lcm
parameters = parameters + parameters_lcm
colors = colors + colors_lcm


# add sd vae decoder
flops = [ flop + 4.98 for flop in flops]
parameters = [ p + 49.49 for p in parameters]


# add lcm ours vae
flops = flops + [7.86 + 0.28]
fid = fid +   [15.79] 
parameters = parameters + [224.6 + 1.22]
colors = colors + ['yellow']


# Create a scatter plot
plt.figure(figsize=(17, 10))
hatch = ["//" ] * len(fid)
scatter = plt.scatter(flops, fid, s=[p * size_scale for p in parameters], alpha=0.7, c=colors ,edgecolors="w",hatch=hatch)



# Add titles and labels
#plt.title('Latency vs Accuracy with Parameter Size')
plt.xlabel('FLOPs (T)', fontsize=20)
plt.ylabel('FID (30K)', fontsize=20)

# Set x and y axis limits
plt.xlim(5, 41)  # Change the range of x-axis
plt.ylim(11, 25)  # Change the range of y-axis

# increase axix fontsize
plt.xticks(fontsize=16) 
plt.yticks(fontsize=16) 

# Set grid color to shallow gray
plt.grid(color='gray', linestyle='--', linewidth=0.5)  
# Show grid
plt.grid(True)

plt.savefig('../figures/scatter.pdf', format='pdf')
# Show the plot
plt.show()




# python3 notebooks/plots/plot_flops_fid.py 
# %%
