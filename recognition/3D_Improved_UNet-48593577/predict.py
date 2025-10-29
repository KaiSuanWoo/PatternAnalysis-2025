import nibabel as nib, numpy as np, torch, torch.nn.functional as F
from modules import UNet3D_Improved

def sliding_window(net, vol, window=(128,128,96), overlap=0.5):
    net.eval()
    C = 6
    D,H,W = vol.shape
    pd,ph,pw = window
    sd,sh,sw = int(pd*(1-overlap)), int(ph*(1-overlap)), int(pw*(1-overlap))
    prob = np.zeros((C,D,H,W), np.float32)
    norm = np.zeros((1,D,H,W), np.float32)

    for z in range(0, max(1, D-pd+1), sd or 1):
        for y in range(0, max(1, H-ph+1), sh or 1):
            for x in range(0, max(1, W-pw+1), sw or 1):
                z0,y0,x0 = z, y, x
                z1,y1,x1 = min(z0+pd,D), min(y0+ph,H), min(x0+pw,W)
                patch = np.zeros((pd,ph,pw), np.float32)
                patch[:z1-z0,:y1-y0,:x1-x0] = vol[z0:z1,y0:y1,x0:x1]
                t = torch.from_numpy(patch[None,None]).cuda()
                with torch.no_grad(), torch.autocast("cuda", torch.float16):
                    logits = net(t)
                    if isinstance(logits, list): logits = logits[0]
                    pr = F.softmax(logits, dim=1)[0].float().cpu().numpy()
                prob[:, z0:z1, y0:y1, x0:x1] += pr[:, :z1-z0, :y1-y0, :x1-x0]
                norm[:, z0:z1, y0:y1, x0:x1] += 1.0
    prob /= np.clip(norm, 1e-6, None)
    return prob.argmax(0).astype(np.uint8)

def run(in_img_path, body_mask_path, out_path):
    vol = nib.load(in_img_path)
    img = vol.get_fdata().astype(np.float32)
    body = nib.load(body_mask_path).get_fdata().astype(np.uint8)
    img = (img - img[body>0].mean()) / (img[body>0].std()+1e-6)

    net = UNet3D_Improved(num_classes=6, deep_supervision=True).cuda()
    net.load_state_dict(torch.load("best_improved_unet3d.pt"))
    lab = sliding_window(net, img)
    nib.Nifti1Image(lab, vol.affine).to_filename(out_path)

# Example:
# run("semantic_MRs/p001.nii.gz", "semantic_labels_only/p001.nii.gz", "preds/p001_pred.nii.gz")
