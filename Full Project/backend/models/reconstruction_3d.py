"""
reconstruction_3d.py — 3D tumour shape reconstruction
Generates interactive Plotly figures + surface mesh via marching cubes
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class TumourReconstructor3D:
    def __init__(self, voxel_spacing=(1.0,1.0,1.0)):
        self.spacing = voxel_spacing

    def mask_to_3d_volume(self, mask2d, depth_slices=30):
        """Extrude 2D mask into a 3D volume using Gaussian depth weighting."""
        if mask2d.ndim == 3: mask2d = mask2d[:,:,0]
        h, w = mask2d.shape
        binary = (mask2d > 127).astype(np.float32)
        if binary.sum() == 0:
            return np.zeros((depth_slices, h, w), dtype=np.float32)

        # Distance transform for smooth depth weighting
        dist = cv2.distanceTransform((binary*255).astype(np.uint8),
                                      cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist = dist / dist.max()

        volume = np.zeros((depth_slices, h, w), dtype=np.float32)
        center = depth_slices // 2
        for z in range(depth_slices):
            t = abs(z - center) / (depth_slices / 2)
            weight = np.exp(-2.5 * t**2)   # Gaussian depth profile
            volume[z] = dist * weight

        return (volume > 0.55).astype(np.float32)

    def volume_to_points(self, volume, max_points=4000):
        dx, dy, dz = self.spacing
        zz, yy, xx = np.where(volume > 0)
        if len(xx) == 0:
            return np.zeros((0,3), np.float32)
        if len(xx) > max_points:
            idx = np.random.choice(len(xx), max_points, replace=False)
            xx, yy, zz = xx[idx], yy[idx], zz[idx]
        return np.stack([xx*dx, yy*dy, zz*dz], axis=1).astype(np.float32)

    def compute_stats(self, volume):
        dx, dy, dz = self.spacing
        n = int(volume.sum())
        if n == 0:
            return {"volume_mm3":0,"num_voxels":0,"centroid_mm":[0,0,0],
                    "extent_slices":0,"bounding_box_mm":{}}
        zz,yy,xx = np.where(volume > 0)
        return {
            "volume_mm3": round(n*dx*dy*dz, 2),
            "num_voxels": n,
            "centroid_mm": [round(float(xx.mean()*dx),2),
                            round(float(yy.mean()*dy),2),
                            round(float(zz.mean()*dz),2)],
            "extent_slices": int(zz.max()-zz.min()+1),
            "bounding_box_mm": {
                "x": [round(float(xx.min()*dx),2), round(float(xx.max()*dx),2)],
                "y": [round(float(yy.min()*dy),2), round(float(yy.max()*dy),2)],
                "z": [round(float(zz.min()*dz),2), round(float(zz.max()*dz),2)],
            }
        }

    def plot_3d_surface(self, volume, title="3D Tumour Reconstruction",
                        tumor_class="Glioma", confidence=0.0):
        """Generate interactive Plotly surface mesh."""
        try:
            import plotly.graph_objects as go
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return self.plot_3d_points(self.volume_to_points(volume), title)

        try:
            from skimage.measure import marching_cubes
            smooth = gaussian_filter(volume.astype(np.float32), sigma=1.2)
            verts, faces, _, _ = marching_cubes(smooth, level=0.35)
        except Exception as e:
            logger.warning("Marching cubes failed: %s — using point cloud", e)
            return self.plot_3d_points(self.volume_to_points(volume), title)

        dx, dy, dz = self.spacing
        x = verts[:,2]*dx; y = verts[:,1]*dy; z = verts[:,0]*dz
        i,j,k = faces[:,0], faces[:,1], faces[:,2]

        # Color map by tumor type
        color_map = {
            "Glioma":     ("#FF4444","Reds"),
            "Meningioma": ("#FF8C00","Oranges"),
            "Pituitary":  ("#FFD700","YlOrBr"),
        }
        _, colorscale = color_map.get(tumor_class, ("#FF4444","Reds"))

        mesh = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=z, colorscale=colorscale, opacity=0.82,
            lighting=dict(ambient=0.4, diffuse=0.85,
                          specular=0.4, roughness=0.3),
            lightposition=dict(x=100, y=200, z=150),
            name="Tumour surface",
            showscale=False,
            hovertemplate="X:%{x:.1f}mm Y:%{y:.1f}mm Z:%{z:.1f}mm<extra></extra>",
        )

        # Wireframe overlay — sample directly from marching-cubes surface verts
        # so the scatter stays tightly bound to the tumour mesh, not the full volume
        n_wire = min(600, len(verts))
        wire_idx = np.random.choice(len(verts), n_wire, replace=False)
        wx = verts[wire_idx, 2] * dx
        wy = verts[wire_idx, 1] * dy
        wz = verts[wire_idx, 0] * dz
        scatter = go.Scatter3d(
            x=wx, y=wy, z=wz,
            mode="markers",
            marker=dict(size=1.5, color="rgba(255,255,255,0.25)"),
            name="Surface cloud",
            hoverinfo="skip",
        )

        fig = go.Figure(data=[mesh, scatter])
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sup>{tumor_class}  |  conf {confidence:.1f}%</sup>",
                x=0.5, font=dict(size=15, color="white"),
            ),
            scene=dict(
                xaxis=dict(title="X (mm)", backgroundcolor="#111827",
                           gridcolor="#374151", color="white"),
                yaxis=dict(title="Y (mm)", backgroundcolor="#111827",
                           gridcolor="#374151", color="white"),
                zaxis=dict(title="Z (mm)", backgroundcolor="#111827",
                           gridcolor="#374151", color="white"),
                bgcolor="#0d1117",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
                aspectmode="data",
            ),
            paper_bgcolor="#0d1117",
            font=dict(color="white", family="monospace"),
            margin=dict(l=0, r=0, t=60, b=0),
            height=520,
            legend=dict(bgcolor="#1f2937", bordercolor="#374151",
                        font=dict(color="white")),
        )
        return fig.to_dict()

    def plot_3d_points(self, points, title="3D Tumour"):
        try:
            import plotly.graph_objects as go
        except ImportError:
            return {}
        if len(points) == 0:
            fig = go.Figure()
            fig.update_layout(title="No tumour detected",
                              paper_bgcolor="#0d1117",
                              font=dict(color="white"))
            return fig.to_dict()
        x,y,z = points[:,0], points[:,1], points[:,2]
        fig = go.Figure(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=2, color=z, colorscale="Hot", opacity=0.75),
        ))
        fig.update_layout(title=title, paper_bgcolor="#0d1117",
                          font=dict(color="white"),
                          scene=dict(bgcolor="#0d1117"),
                          margin=dict(l=0,r=0,t=40,b=0), height=520)
        return fig.to_dict()