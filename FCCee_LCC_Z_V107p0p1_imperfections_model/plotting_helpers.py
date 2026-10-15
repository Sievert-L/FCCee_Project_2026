# Plotting helpers for the optics after correction (notebook 04)
#   plot_ring_optics: orbit, beta-beating, phase, coupling RDTs, dispersion and W functions around the ring
#   plot_ip_optics:   the same quantities at the IPs

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Statistics shown in the legend
def rms_s(y, s):
    return np.sqrt(np.trapezoid(y**2, s) / (s[-1] - s[0]))

STATS = {
    'rms':  ('RMS',  lambda y, s: rms_s(y, s)),
    'std':  ('STD',  lambda y, s: np.std(y)),
    'mean': ('Mean', lambda y, s: np.mean(y)),
}

# ============================================================
# Around-the-ring panels: rows of (x-panel, y-panel); each = (ylabel, fn(t, ref), stat)
RING_PANELS = [
    [(r'x ($\mu$m)', lambda t, r: t.x*1e6, 'rms'),
     (r'y ($\mu$m)', lambda t, r: t.y*1e6, 'rms')],
    [(r'$\Delta\beta_x/\beta_x$ (%)', lambda t, r: (t.betx/r.betx - 1)*100, 'rms'),
     (r'$\Delta\beta_y/\beta_y$ (%)', lambda t, r: (t.bety/r.bety - 1)*100, 'rms')],
    [(r'$\Delta\phi_x$ [$10^{-4}$ 2$\pi$]', lambda t, r: (t.mux - r.mux)*1e4, 'rms'),
     (r'$\Delta\phi_y$ [$10^{-4}$ 2$\pi$]', lambda t, r: (t.muy - r.muy)*1e4, 'rms')],
    [(r'$|f_{1001}|$ ($10^{-4}$)', lambda t, r: np.abs(t.f1001)*1e4, 'rms'),
     (r'$|f_{1010}|$ ($10^{-4}$)', lambda t, r: np.abs(t.f1010)*1e4, 'rms')],
    [(r'$\Delta D_x$ (mm)', lambda t, r: (t.dx - r.dx)*1e3, 'rms'),
     (r'$\Delta D_y$ (mm)', lambda t, r: (t.dy - r.dy)*1e3, 'rms')],
    [(r'$\Delta W_x$', lambda t, r: t.wx_chrom - r.wx_chrom, 'rms'),
     (r'$\Delta W_y$', lambda t, r: t.wy_chrom - r.wy_chrom, 'rms')],
]

def plot_ring_optics(tws, tw_ref, panels=RING_PANELS, figsize=(10, 10), fontsize=12):
    fig, axs = plt.subplots(len(panels), 2, figsize=figsize, sharex=True, squeeze=False)
    for axs_row, panel_row in zip(axs, panels):
        for ax, (ylabel, fn, stat) in zip(axs_row, panel_row):
            stat_name, stat_fn = STATS[stat]
            ax.axhline(0, lw=1, c='k')
            for label, tw in tws.items():
                y = np.asarray(fn(tw, tw_ref))
                ax.plot(tw.s, y, lw=0.8, alpha=0.8,
                        label=f'{label}: {stat_name}={stat_fn(y, tw.s):.2f}')
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.legend(fontsize=fontsize-2, framealpha=1, loc='best')
            ax.grid(alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel('s (m)', fontsize=fontsize)
    fig.tight_layout()
    return fig, axs

# ============================================================
# IP panels: same format, no stat. The reference fn(ref, ref) is drawn in black.
IP_PANELS = [
    [(r'x ($\mu$m)', lambda t, r: t.x*1e6),
     (r'y ($\mu$m)', lambda t, r: t.y*1e6)],
    [(r'$\beta_x^*$ (cm)', lambda t, r: t.betx*1e2),
     (r'$\beta_y^*$ (mm)', lambda t, r: t.bety*1e3)],
    [(r'$\alpha_x^*$ ($10^{-3}$)', lambda t, r: t.alfx*1e3),
     (r'$\alpha_y^*$ ($10^{-3}$)', lambda t, r: t.alfy*1e3)],
    [(r'$|f_{1001}|$ ($10^{-4}$)', lambda t, r: np.abs(t.f1001)*1e4),
     (r'$|f_{1010}|$ ($10^{-4}$)', lambda t, r: np.abs(t.f1010)*1e4)],
    [(r'$D_x - D_{x,0}$ ($\mu$m)', lambda t, r: (t.dx - r.dx)*1e6),
     (r'$D_y^*$ ($\mu$m)',         lambda t, r: t.dy*1e6)],
    [(r'$\Delta W_x$', lambda t, r: t.wx_chrom - r.wx_chrom),
     (r'$\Delta W_y$', lambda t, r: t.wy_chrom - r.wy_chrom)],
]

def plot_ip_optics(tws, tw_ref, ips, ip_labels=None, panels=IP_PANELS,
                   figsize=(8, 10), fontsize=12):
    ip_labels = ip_labels if ip_labels is not None else ips
    xpos = np.arange(len(ips))
    ref = tw_ref.rows[ips]
    fig, axs = plt.subplots(len(panels), 2, figsize=figsize, sharex=True, squeeze=False)
    for axs_row, panel_row in zip(axs, panels):
        for ax, (ylabel, fn) in zip(axs_row, panel_row):
            ax.plot(xpos, fn(ref, ref), 'k_', ms=20, mew=1.5, label='reference')
            for label, tw in tws.items():
                ax.plot(xpos, fn(tw.rows[ips], ref), 'o-', alpha=0.8, label=label)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=fontsize-2, framealpha=1)
    for ax in axs[-1]:
        ax.set_xticks(xpos)
        ax.set_xticklabels(ip_labels, fontsize=fontsize)
    fig.tight_layout()
    return fig, axs
