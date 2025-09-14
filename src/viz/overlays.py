import cv2, numpy as np

def overlay_heatmap(gray, cam):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    heat = cv2.applyColorMap((255*cam).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = (0.5*base + 0.5*heat).astype(np.uint8)
    return out