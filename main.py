import os                       
import urllib.request           
import cv2                       
import numpy as np                 
import matplotlib.pyplot as plt    
import matplotlib.gridspec as gridspec
from scipy.ndimage import convolve    
from scipy.optimize import curve_fit  

DW_DAYS  = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)  
DW_CK    = np.array([10, 11, 14, 17, 23, 27, 30, 35], dtype=float)  
DW_MUSIC = np.array([10, 11, 14, 18, 24, 29, 33, 38], dtype=float)  

RAD_DAYS    = np.arange(1, 15, dtype=float)
RAD_CONTROL = np.array([0, 0, 0,  3,  58, 112, 173, 247,
                         342, 436, 527, 633, 727, 813], dtype=float)  
RAD_MUSIC   = np.array([0, 0, 4, 57, 124, 193, 267, 359,
                         451, 543, 658, 761, 872, 984], dtype=float)  

CHICK_DAYS    = np.array([2,  4,  6], dtype=float)   
CHICK_CONTROL = np.array([2,  7, 12], dtype=float)   
CHICK_MUSIC   = np.array([3, 16, 23], dtype=float)   

FIGURE_URL  = ("https://cdn.ncbi.nlm.nih.gov/pmc/blobs/1974/9839374/"
               "8804ab9bb00c/KPSB_A_2163346_F0001_OC.jpg")

_HERE        = os.path.dirname(os.path.abspath(__file__))  
OXFORD_FILE  = os.path.join(_HERE, "oxford.png")           
FIGURE_FILE  = os.path.join(_HERE, "duckweed_figure.jpg")  

BLUE, RED = "steelblue", "tomato"  # control vs music 

# METHOD 1: SOBEL EDGE DETECTION
def sobel(gray):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)
    return np.hypot(convolve(gray.astype(float), Kx),
                    convolve(gray.astype(float), Ky))


def green_area(bgr):
    """Count strong edge pixels inside the plant's green region."""

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
    edges = sobel(mask.astype(float))

    # Count only the STRONG edges -- those above 10% of the strongest edge in this image.
    return int((edges > edges.max() * 0.1).sum())

# METHOD 2: FINITE DIFFERENCES
def diff1(y, t):# y = measured values (e.g. frond counts), t = time points (days)
    
    d = np.empty_like(y, dtype=float)  # create an empty array the same size as y
    d[0]    = (y[1]  - y[0])  / (t[1]  - t[0])
    d[-1]   = (y[-1] - y[-2]) / (t[-1] - t[-2])
    d[1:-1] = (y[2:] - y[:-2]) / (t[2:]  - t[:-2])

    return d  

def diff2(y, t):
    return diff1(diff1(y, t), t)

# METHOD 3: LOGISTIC GROWTH MODEL (NONLINEAR LEAST SQUARES)
#   K  = carrying capacity: the maximum size the plant will EVER reach. If K is larger for the music group, music raised the ceiling.
#   r  = growth rate: how steep the S-curve is. Large r = fast growth. Small r = slow, gradual growth.
#   t0 = inflection point: the day when growth is fastest (middle of the S). This is also what Newton-Raphson confirms in Method 4.
def logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

def fit_logistic(t, y):
    """Fit the logistic curve to data. Returns (best parameters, their errors) or (None, None)."""
    mask = y > 0
    if mask.sum() < 4:
        return None, None  

    t_f, y_f = t[mask], y[mask]  # filtered time and measurement arrays
    p0 = [y_f.max() * 1.2, 0.6, float(t_f[len(t_f) // 2])]

    try:
        popt, pcov = curve_fit(logistic, t_f, y_f, p0=p0, maxfev=30_000,
                               bounds=([0, 1e-4, -50], [5_000, 20, 50]))
        # popt = the best-fit values of [K, r, t0]
        # pcov = covariance matrix. Its diagonal tells us how uncertain each parameter is.
        # np.sqrt(np.diag(pcov)) converts this to standard deviations (the ± errors).
        return popt, np.sqrt(np.diag(pcov))

    except RuntimeError:
        return None, None

# METHOD 4: NEWTON-RAPHSON ROOT-FINDING
def newton(f, df, x0, tol=1e-8, max_iter=50):
    x = float(x0)  
    for i in range(max_iter):
        fx  = f(x)   
        dfx = df(x)  
        if abs(dfx) < 1e-14:
            break

        step = fx / dfx  # how far to move 
        x -= step        # move to the new guess
        if abs(step) < tol:
            return x, i + 1
    return x, max_iter


def peak_growth_day(popt):
    """Find the exact day of fastest growth by solving h''(t) = 0 with Newton-Raphson."""

    K, r, t0 = popt  

    h2 = lambda t: (K*r**2 * np.exp(-r*(t-t0)) * (np.exp(-r*(t-t0)) - 1)
                    / (1 + np.exp(-r*(t-t0)))**3)

    h3 = lambda t: (K*r**3 * np.exp(-r*(t-t0))
                    * (np.exp(-r*(t-t0))**2 - 4*np.exp(-r*(t-t0)) + 1)
                    / (1 + np.exp(-r*(t-t0)))**4)
    return newton(h2, h3, t0 + 0.5)


def load_duckweed_figure():
    if not os.path.exists(FIGURE_FILE):
        print(f"  Downloading {FIGURE_FILE} ...")
        urllib.request.urlretrieve(FIGURE_URL, FIGURE_FILE)
    return cv2.imread(FIGURE_FILE)  

def duckweed_green_areas(img):
    """Split the duckweed figure into 6 panels (2 rows x 3 columns) and measure green area in each."""

    H, W = img.shape[:2]
    area = img[:int(H * 0.53), int(W * 0.09):]

    ph, pw = area.shape[:2]  
    rh = ph // 2   
    cw = pw // 3   
    pad = int(min(rh, cw) * 0.04)  

    areas_ck, areas_mu = [], []  
    for ri, bucket in enumerate([areas_ck, areas_mu]):
        for ci in range(3):
            panel = area[ri*rh+pad:(ri+1)*rh-pad, ci*cw+pad:(ci+1)*cw-pad]
            bucket.append(green_area(panel))  
    return np.array(areas_ck, float), np.array(areas_mu, float)

def oxford_green_areas(img):
    """Split oxford.png into 3 horizontal bands (Control / Classical / Rock) and count edge pixels."""

    H, W = img.shape[:2]  
    crop = img[int(H * 0.10):, int(W * 0.10):]
    H2 = crop.shape[0]   
    bh = H2 // 3         
    areas = []
    for i in range(3):  
        band = crop[i*bh:(i+1)*bh, :]               
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)  
        edges = sobel(gray)                           
        areas.append(int((edges > 20).sum()))

    return np.array(areas, float)

def plot_all(dw_green_ck, dw_green_mu, ox_green,
             dw_r1_ck, dw_r1_mu,
             dw_popt_ck, dw_popt_mu,
             rad_r1_ck, rad_r1_mu, rad_popt_ck, rad_popt_mu,
             chick_r1_ck, chick_r1_mu,
             peaks):
    fig = plt.figure(figsize=(18, 14))

    fig.suptitle("Music's Effect on Plant Growth -- CP1\n"
                 "Methods: Sobel | Finite Differences | Logistic Fit | Newton-Raphson",
                 fontsize=12, fontweight="bold")
  
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.50, wspace=0.38)

    for col, (days, ck, mu, pck, pmu, ylabel, title) in enumerate([
        (DW_DAYS,  DW_CK,       DW_MUSIC,   dw_popt_ck,  dw_popt_mu,
         "Frond count", "Duckweed -- Logistic Fit"),
        (RAD_DAYS, RAD_CONTROL, RAD_MUSIC,  rad_popt_ck, rad_popt_mu,
         "Height (mm)",  "Radish -- Logistic Fit"),
    ]):
        ax = fig.add_subplot(gs[0, col*2 : col*2+2]) 
        ax.scatter(days, ck, color=BLUE, s=45, zorder=5, label="Control/CK")  
        ax.scatter(days, mu, color=RED,  s=45, zorder=5, label="Music")
        t_fine = np.linspace(days[0], days[-1]+1, 300)

        if pck is not None:
            ax.plot(t_fine, logistic(t_fine, *pck), BLUE, lw=2,
                    label=f"K={pck[0]:.1f}")  
        if pmu is not None:
            ax.plot(t_fine, logistic(t_fine, *pmu), RED, lw=2,
                    label=f"K={pmu[0]:.1f}")

        ax.set_xlabel("Day"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n(Method 3: nonlinear least squares)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)  

    ax_c = fig.add_subplot(gs[0, 4:])  
    x = np.arange(3)  
    ax_c.bar(x - 0.18, CHICK_CONTROL, 0.35, color=BLUE, alpha=0.85, label="Control")
    ax_c.bar(x + 0.18, CHICK_MUSIC,   0.35, color=RED,  alpha=0.85, label="Music")
    ax_c.set_xticks(x); ax_c.set_xticklabels(["Day 2", "Day 4", "Day 6"])
    ax_c.set_ylabel("Saplings sprouted")
    ax_c.set_title("Chickpea -- Germination\n(Chowdhury & Gupta)")
    ax_c.legend(fontsize=7); ax_c.grid(alpha=0.3, axis="y")

    rate_sets = [
        (DW_DAYS,    dw_r1_ck,    dw_r1_mu,    "dN/dt (fronds/day)", "Duckweed",
         "Duckweed CK", "Duckweed Music"),
        (RAD_DAYS,   rad_r1_ck,   rad_r1_mu,   "dH/dt (mm/day)",     "Radish",
         "Radish Control", "Radish Music"),
        (CHICK_DAYS, chick_r1_ck, chick_r1_mu, "Saplings/day",       "Chickpea",
         None, None),  
    ]
    for col, (days, r_ck, r_mu, ylabel, title, k_ck, k_mu) in enumerate(rate_sets):
        ax = fig.add_subplot(gs[1, col*2 : col*2+2])
        ax.plot(days, r_ck, color=BLUE, lw=2, marker="o", ms=5, label="Control/CK")
        ax.plot(days, r_mu, color=RED,  lw=2, marker="o", ms=5, label="Music")

        ax.fill_between(days, r_ck, alpha=0.12, color=BLUE)
        ax.fill_between(days, r_mu, alpha=0.12, color=RED)

        if k_ck and k_ck in peaks:
            ax.axvline(peaks[k_ck], color=BLUE, lw=1.5, ls="--",
                       label=f"Peak: day {peaks[k_ck]:.2f}")
        if k_mu and k_mu in peaks:
            ax.axvline(peaks[k_mu], color=RED, lw=1.5, ls="--",
                       label=f"Peak: day {peaks[k_mu]:.2f}")

        ax.set_xlabel("Day"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title} -- Growth Rate\n(Method 2: finite differences)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax_dg = fig.add_subplot(gs[2, :2])  
    x = np.arange(3)  
    ax_dg.bar(x - 0.18, dw_green_ck/1e3, 0.35, color=BLUE, alpha=0.8, label="CK")
    ax_dg.bar(x + 0.18, dw_green_mu/1e3, 0.35, color=RED,  alpha=0.8, label="Music")
    ax_dg.set_xticks(x); ax_dg.set_xticklabels(["Day 0", "Day 4", "Day 7"])
    ax_dg.set_ylabel("Sobel edge pixels")  
    ax_dg.set_title("Duckweed -- Plant Boundary Size\n(Method 1: Sobel edge detection)")
    ax_dg.legend(fontsize=7); ax_dg.grid(alpha=0.3, axis="y")

    ax_ox = fig.add_subplot(gs[2, 2:4])  
    labels = ["Control", "Classical", "Rock"]
    colors = [BLUE, RED, "mediumseagreen"] 
    ax_ox.bar(labels, ox_green/1e3, color=colors, alpha=0.85)
    ax_ox.set_ylabel("Sobel edge pixels")
    ax_ox.set_title("Bok Choy -- Plant Boundary by Condition\n(Method 1: Sobel on oxford.png)")
    ax_ox.grid(alpha=0.3, axis="y")
    ax_tbl = fig.add_subplot(gs[2, 4:])  
    ax_tbl.axis("off")  

    rows = []  
    for lbl, popt, pk in [("Duckweed CK",    dw_popt_ck,  "Duckweed CK"),
                           ("Duckweed Music", dw_popt_mu,  "Duckweed Music"),
                           ("Radish Control", rad_popt_ck, "Radish Control"),
                           ("Radish Music",   rad_popt_mu, "Radish Music")]:
        if popt is not None:
            # popt[0]=K, popt[1]=r, popt[2]=t0 -- formatted to a readable number of decimals
            rows.append([lbl, f"{popt[0]:.1f}", f"{popt[1]:.4f}",
                         f"{popt[2]:.2f}", f"{peaks.get(pk, float('nan')):.2f}"])

    tbl = ax_tbl.table(cellText=rows,
                       colLabels=["Group", "K", "r", "t0", "Peak day"],
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)  
    tbl.set_fontsize(8)            
    tbl.scale(1, 1.7)              
    ax_tbl.set_title("Logistic Parameters + Peak Growth Days\n"
                     "(Methods 3+4: curve fit + Newton-Raphson)", fontsize=9)

    plt.savefig("plant_growth_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: plant_growth_analysis.png")

def main():
    print("=" * 55)
    print("CP1 -- Music's Effect on Plant Growth")
    print("=" * 55)
    print("\n[1/4] Sobel edge detection -- measuring green area ...")

    dw_img = load_duckweed_figure()

    dw_green_ck, dw_green_mu = duckweed_green_areas(dw_img)
    print(f"  Duckweed CK    Sobel edge px: {dw_green_ck.astype(int)}")
    print(f"  Duckweed Music Sobel edge px: {dw_green_mu.astype(int)}")
    print(f"  oxford path: {OXFORD_FILE}")
    print(f"  oxford exists: {os.path.exists(OXFORD_FILE)}")
    ox_img = cv2.imread(OXFORD_FILE)
    if ox_img is None:
        print(f"  WARNING: could not load oxford.png")
        ox_green = np.zeros(3)
    else:
        ox_green = oxford_green_areas(ox_img)
        print(f"  Bok choy Sobel edge px -- Control/Classical/Rock: {ox_green.astype(int)}")
    print("\n[2/4] Central finite differences ...")

    dw_r1_ck    = diff1(DW_CK,         DW_DAYS)   # duckweed control growth rate
    dw_r1_mu    = diff1(DW_MUSIC,      DW_DAYS)   # duckweed music growth rate
    rad_r1_ck   = diff1(RAD_CONTROL,   RAD_DAYS)  # radish control growth rate
    rad_r1_mu   = diff1(RAD_MUSIC,     RAD_DAYS)  # radish music growth rate
    chick_r1_ck = diff1(CHICK_CONTROL, CHICK_DAYS)  # chickpea control germination rate
    chick_r1_mu = diff1(CHICK_MUSIC,   CHICK_DAYS)  # chickpea music germination rate

    print("\n[3/4] Fitting logistic growth models ...")

    dw_popt_ck,  _ = fit_logistic(DW_DAYS,  DW_CK)       # duckweed control fit
    dw_popt_mu,  _ = fit_logistic(DW_DAYS,  DW_MUSIC)    # duckweed music fit
    rad_popt_ck, _ = fit_logistic(RAD_DAYS, RAD_CONTROL) # radish control fit
    rad_popt_mu, _ = fit_logistic(RAD_DAYS, RAD_MUSIC)   # radish music fit
    for name, p in [("Duckweed CK",    dw_popt_ck),
                    ("Duckweed Music", dw_popt_mu),
                    ("Radish Control", rad_popt_ck),
                    ("Radish Music",   rad_popt_mu)]:
        if p is not None:
            # p[0]=K (max size), p[1]=r (growth rate), p[2]=t0 (inflection day)
            print(f"  {name:20s}: K={p[0]:.2f}  r={p[1]:.4f}  t0={p[2]:.2f}")
    print("\n[4/4] Newton-Raphson root-finding ...")

    peaks = {} 

    for name, popt in [("Duckweed CK",    dw_popt_ck),
                       ("Duckweed Music", dw_popt_mu),
                       ("Radish Control", rad_popt_ck),
                       ("Radish Music",   rad_popt_mu)]:
        if popt is not None:
            day, iters = peak_growth_day(popt)
            peaks[name] = day  
            print(f"  {name:20s}: peak day {day:.3f}  ({iters} iterations)")

    plot_all(dw_green_ck, dw_green_mu, ox_green,
             dw_r1_ck, dw_r1_mu,
             dw_popt_ck, dw_popt_mu,
             rad_r1_ck, rad_r1_mu, rad_popt_ck, rad_popt_mu,
             chick_r1_ck, chick_r1_mu, peaks)
    print("Done.")

if __name__ == "__main__":
    main()
