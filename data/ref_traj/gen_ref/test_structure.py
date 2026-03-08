import scipy.io as scio
data = scio.loadmat('MPC/ref_trajectory/snake_trajectory_ref.mat', squeeze_me=True)
print("Keys:", list(data.keys()))
if 'ref_trajectory' in data:
    print("ref_trajectory type:", type(data['ref_trajectory']))
    print("ref_trajectory shape:", getattr(data['ref_trajectory'], 'shape', 'N/A'))