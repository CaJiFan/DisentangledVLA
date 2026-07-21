import torch

stats = torch.load("./checkpoints/text_tcvae/libero_spatial/dataset_statistics.pt")
# Look at the raw min and max of the action space
print("Min:", stats['libero_spatial_no_noops']['action']['min'])
print("Max:", stats['libero_spatial_no_noops']['action']['max'])


