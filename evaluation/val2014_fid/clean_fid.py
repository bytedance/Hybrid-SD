from cleanfid import fid

ours_path="/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/results/HybridSD_dpm_guidance7_fuxian_generate/im256"
orig_path="/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_41k_full/val2014_30K_256"

score = fid.compute_fid(ours_path, orig_path, mode="clean", dataset_res=256, batch_size=128)
print(score)