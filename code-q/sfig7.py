import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter, FuncFormatter

import geopandas as gpd
from shapely.ops import unary_union
from shapely import vectorized

from scipy.stats import linregress, norm, t


# ================== 0) Paths (codes/ -> project root) ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../codes
PROJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))       # project root
DATA_DIR = os.path.join(PROJ_DIR, "data")
FIG_DIR  = os.path.join(PROJ_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)

# ================== 0.1) Font: Arial (from data/fonts/Arial; fallback if missing) ==================
FONT_DIR = os.path.join(DATA_DIR, "fonts", "Arial")
for fn in ["ARIAL.TTF", "ARIALBD.TTF", "ARIALI.TTF", "ARIALBI.TTF",
           "Arial.ttf", "Arialbd.ttf", "Ariali.ttf", "Arialbi.ttf"]:
    p = os.path.join(FONT_DIR, fn)
    if os.path.isfile(p):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"


# =============================================================================
# Input root (IMPORTANT): both maps and time series read from ../data/sfig7_data
# =============================================================================
ROOT_DIR_INPUT = os.path.join(DATA_DIR, "sfig7_data") 


# =============================================================================
# Part A: Fig.4(a–c) cumulative exposure maps
# =============================================================================

# ------------------ Colormap ------------------
white_red = LinearSegmentedColormap.from_list("white_red", ["white", "#3B83B4"])
white_red.set_bad("white")  # show NaN (masked C100) as white

# ------------------ Input configuration ------------------
regions = ["East_Asia", "South_Asia", "USA"]
ranges_ = ["500km"]
PERIODS = {"2000-2023": range(2000, 2024)}  # inclusive end -> 2023

region_extent = {
    "East_Asia":  [100, 133, 7, 33],
    "South_Asia": [74, 96, 6, 30],
    "USA":        [-103, -69, 10, 40],
}

region_ticks = {
    "East_Asia":  {"xticks": list(range(105, 133, 10)),  "yticks": list(range(10, 33, 5))},
    "South_Asia": {"xticks": [75, 85, 95],               "yticks": [10, 15, 20, 25]},
    "USA":        {"xticks": list(range(-100, -69, 10)), "yticks": [10, 15, 20, 25, 30, 35]},
}

var_name = "pop_rain_exposure"
VMAX = 50000.0
VMIN = 0.0

TICK_LABELSIZE = 26
FIG_DPI = 400

# ------------------ Load C100 shapefile (once) ------------------
C100_SHP = os.path.join(DATA_DIR, "shp", "c100.shp")
if not os.path.isfile(C100_SHP):
    raise FileNotFoundError(f"c100.shp not found: {C100_SHP}")
c100_gdf = gpd.read_file(C100_SHP).to_crs(epsg=4326)
c100_union = unary_union(c100_gdf.geometry)

def mask_c100_to_nan(da: xr.DataArray, geom_union):
    """Set pixels inside C100 to NaN so they are not displayed."""
    lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
    lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Cannot identify lon/lat coordinate names: coords={list(da.coords)}")

    lon = da[lon_name].values
    lat = da[lat_name].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    inside = vectorized.contains(geom_union, lon2d, lat2d)

    mask_da = xr.DataArray(
        inside,
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
        dims=(lat_name, lon_name),
    )
    return da.where(~mask_da)

def save_shared_colorbar(out_path: str):
    """Save ONE shared horizontal colorbar (same cmap/vmin/vmax/label for all maps)."""
    fig = plt.figure(figsize=(15, 1.2), dpi=FIG_DPI)
    ax = fig.add_axes([0.08, 0.35, 0.84, 0.35])

    norm_ = Normalize(vmin=VMIN, vmax=VMAX)
    sm = cm.ScalarMappable(norm=norm_, cmap=white_red)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal", extend="max", extendfrac=0.06)
    cbar.set_label("Exposure to TC rainfall", fontsize=TICK_LABELSIZE)
    cbar.ax.tick_params(which="major", direction="out", length=8, width=1.2, labelsize=TICK_LABELSIZE)

    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.xaxis.set_major_formatter(cbar.formatter)
    cbar.update_ticks()

    offset_text = cbar.ax.xaxis.get_offset_text()
    offset_text.set_size(TICK_LABELSIZE)
    offset_text.set_fontname("Arial")

    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# ------------------ Map output name mapping (as requested) ------------------
MAP_OUTNAME = {
    "USA":        "sfig7a.png",
    "South_Asia": "sfig7b.png",
    "East_Asia":  "sfig7c.png",
}

# ------------------ Run map accumulation & plotting ------------------
for region in regions:
    lon_min, lon_max, lat_min, lat_max = region_extent[region]
    xticks = region_ticks[region]["xticks"]
    yticks = region_ticks[region]["yticks"]

    for drange in ranges_:
        data_dir = os.path.join(ROOT_DIR_INPUT, region, drange, "pop_exposure")
        if not os.path.isdir(data_dir):
            print(f"[INFO] Path does not exist, skip: {data_dir}")
            continue

        for period_name, year_list in PERIODS.items():
            if period_name != "2000-2023":
                continue

            sum_da = None
            has_data = False

            for year in year_list:
                nc_path = os.path.join(data_dir, f"pop_exposure_{year}.nc")
                if not os.path.isfile(nc_path):
                    print(f"  [MISSING] {nc_path} (skip this year)")
                    continue

                try:
                    with xr.open_dataset(nc_path) as ds:
                        da = ds[var_name]
                        da_filled = da.fillna(0)
                        sum_da = da_filled.copy() if sum_da is None else (sum_da + da_filled)
                        has_data = True
                except Exception as e:
                    print(f"  [ERROR] Failed to open {nc_path}: {e}")

            if not has_data or sum_da is None:
                print(f"[WARN] {region} {drange} {period_name}: no usable data, skip plotting")
                continue

            # mask out C100 region
            sum_da = mask_c100_to_nan(sum_da, c100_union)

            # plot map (no colorbar)
            proj = ccrs.PlateCarree()
            fig = plt.figure(figsize=(9, 6), dpi=FIG_DPI)
            ax = plt.axes(projection=proj)

            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

            ax.set_xticks(xticks, crs=proj)
            ax.set_yticks(yticks, crs=proj)
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())

            ax.xaxis.set_tick_params(which="major", direction="out", length=8, width=1.2,
                                     labelsize=TICK_LABELSIZE, bottom=True, top=False)
            ax.yaxis.set_tick_params(which="major", direction="out", length=8, width=1.2,
                                     labelsize=TICK_LABELSIZE, left=True, right=False)
            ax.minorticks_off()
            ax.spines["geo"].set_linewidth(1.2)

            # draw C100 boundary dashed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c100_gdf.boundary.plot(ax=ax, transform=proj, color="black",
                                       linewidth=1.0, linestyle="--", zorder=6)

            _ = sum_da.plot.pcolormesh(
                ax=ax, transform=proj, cmap=white_red,
                vmin=VMIN, vmax=VMAX,
                add_colorbar=False, add_labels=False
            )
            ax.set_title("")

            out_png = os.path.join(FIG_DIR, MAP_OUTNAME[region])
            plt.tight_layout()
            fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)

# shared colorbar
shared_cbar_path = os.path.join(FIG_DIR, "sfig7_colorbar.png")
save_shared_colorbar(shared_cbar_path)


# =============================================================================
# Part B: Fig.4(d–f) total exposure time series outside C100 + OLS + 95% CI
# =============================================================================

years = range(2000, 2024)
ranges_ts = ["500km"]
region_label = {"East_Asia": "WNP", "South_Asia": "BOB", "USA": "WNA"}

TS_OUTNAME = {
    "USA":        "sfig7d.png",
    "South_Asia": "sfig7e.png",
    "East_Asia":  "sfig7f.png",
}

# y-axis unify controls
MANUAL_YAXIS = True
Y_EXP = 7
Y_LIM_SCALED = (-0.2, 2.5)
Y_TICKS_SCALED = [0.0, 1.0, 2.0]
AUTO_PAD_RATIO = 1.05
AUTO_STEP_SCALED = 0.5

# font sizes
TICK_FONTSIZE = 18
LABEL_FONTSIZE = 18
ANN_FONTSIZE = 14
EXP_FONTSIZE = 18

TEXT_POS = {
    "East_Asia":  (0.33, 0.98, 0.06),
    "South_Asia": (0.33, 0.98, 0.06),
    "USA":        (0.33, 0.98, 0.06),
}

def mann_kendall_pvalue(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3:
        return np.nan

    S = 0
    for i in range(n - 1):
        S += np.sum(np.sign(y[i + 1:] - y[i]))

    unique, counts = np.unique(y, return_counts=True)
    tie_counts = counts[counts > 1]

    varS = (n * (n - 1) * (2 * n + 5)) / 18.0
    if tie_counts.size > 0:
        varS -= np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)) / 18.0
    if varS <= 0:
        return np.nan

    if S > 0:
        Z = (S - 1) / np.sqrt(varS)
    elif S < 0:
        Z = (S + 1) / np.sqrt(varS)
    else:
        Z = 0.0

    return 2 * (1 - norm.cdf(abs(Z)))

def build_mask_outside_c100(lat_vals, lon_vals, geom_union):
    Lon, Lat = np.meshgrid(lon_vals, lat_vals)  # [lat, lon]
    inside = vectorized.contains(geom_union, Lon, Lat)
    return ~inside  # True = keep

def apply_common_yaxis(ax, exp, y_lim_scaled, y_ticks_scaled,
                       tick_fs=16, exp_fs=12, fontname="Arial"):
    scale = 10 ** exp
    ax.set_ylim(y_lim_scaled[0] * scale, y_lim_scaled[1] * scale)
    ax.set_yticks(np.array(y_ticks_scaled, dtype=float) * scale)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/scale:g}"))
    ax.tick_params(axis="y", labelsize=tick_fs)

    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(0.0, 1.01, rf"$\times 10^{{{exp}}}$",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=exp_fs, fontname=fontname)

# ------------------ First pass: compute series and trend params ------------------
all_results = {}  # region -> dict(series, fit_params, y_max)

for region in regions:
    if region not in region_label:
        continue

    range_series = {}
    range_fit_params = {}
    mask_da = None

    for drange in ranges_ts:
        data_dir = os.path.join(ROOT_DIR_INPUT, region, drange, "pop_exposure")
        if not os.path.isdir(data_dir):
            print(f"  [INFO] Path does not exist, skip: {data_dir}")
            continue

        # build mask once using a sample file
        if mask_da is None:
            sample_nc = None
            for yy in years:
                test_path = os.path.join(data_dir, f"pop_exposure_{yy}.nc")
                if os.path.isfile(test_path):
                    sample_nc = test_path
                    break
            if sample_nc is None:
                print(f"  [WARN] {region} {drange} has no nc files, skip")
                continue

            with xr.open_dataset(sample_nc) as ds_sample:
                da_sample = ds_sample[var_name]
                if da_sample.dims[0] != "lat":
                    da_sample = da_sample.transpose("lat", "lon")
                lat_vals = da_sample["lat"].values
                lon_vals = da_sample["lon"].values

            mask_np = build_mask_outside_c100(lat_vals, lon_vals, c100_union)
            mask_da = xr.DataArray(mask_np, coords={"lat": lat_vals, "lon": lon_vals}, dims=("lat", "lon"))

        totals, year_list = [], []
        for y in years:
            nc_path = os.path.join(data_dir, f"pop_exposure_{y}.nc")
            if not os.path.isfile(nc_path):
                totals.append(0.0)
                year_list.append(y)
                continue

            try:
                with xr.open_dataset(nc_path) as ds:
                    da = ds[var_name]
                    if da.dims[0] != "lat":
                        da = da.transpose("lat", "lon")

                    da_clip = da.where(mask_da, other=0.0)
                    total_val = float(da_clip.fillna(0).sum().values)

                totals.append(total_val)
                year_list.append(y)
            except Exception as e:
                print(f"  [ERROR] Failed to open {nc_path}: {e}, set to 0 for this year")
                totals.append(0.0)
                year_list.append(y)

        series = pd.Series(data=totals, index=year_list, name=drange)
        range_series[drange] = series

        x_year = np.array(year_list, dtype=float)
        y_val = np.array(totals, dtype=float)

        slope, intercept, rvalue, p_lr, stderr = linregress(x_year, y_val)
        p_mk = mann_kendall_pvalue(y_val)
        range_fit_params[drange] = (slope, intercept, rvalue, p_mk, stderr)

    if not range_series:
        print(f"[INFO] {region_label[region]} has no valid data, skip this region.")
        continue

    y_max = max(float(np.nanmax(s.values)) for s in range_series.values())
    all_results[region] = {"range_series": range_series, "fit_params": range_fit_params, "y_max": y_max}

if not all_results:
    raise RuntimeError("No valid results produced. Please check input paths and files.")

# ------------------ Decide common y-axis ------------------
if MANUAL_YAXIS:
    common_exp = Y_EXP
    common_ylim_scaled = Y_LIM_SCALED
    common_yticks_scaled = Y_TICKS_SCALED
else:
    common_exp = Y_EXP
    scale = 10 ** common_exp
    y_max_common = max(v["y_max"] for v in all_results.values()) * AUTO_PAD_RATIO
    top_scaled = np.ceil((y_max_common / scale) / AUTO_STEP_SCALED) * AUTO_STEP_SCALED
    common_ylim_scaled = (0.0, float(top_scaled))
    common_yticks_scaled = list(np.arange(0.0, top_scaled + 0.5 * AUTO_STEP_SCALED, AUTO_STEP_SCALED))


# ------------------ Second pass: plotting ------------------
color_map_range = {"500km": "#3B83B4"}

for region in regions:
    if region not in all_results:
        continue

    range_series = all_results[region]["range_series"]
    fit_params = all_results[region]["fit_params"]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=FIG_DPI)

    for i, (drange, series) in enumerate(sorted(range_series.items())):
        x = np.array(series.index, dtype=float)
        y = series.values.astype(float)
        color = color_map_range.get(drange, "#3B83B4")

        # observations
        ax.plot(x, y, marker="o", linestyle="-", linewidth=2.4, markersize=4,
                color=color, zorder=3)

        # OLS fit + 95% CI
        slope, intercept, rvalue, p_mk, stderr = fit_params[drange]
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = intercept + slope * x_line

        n = len(x)
        if n >= 3:
            y_hat = intercept + slope * x
            resid = y - y_hat
            dof = n - 2
            s_err = np.sqrt(np.sum(resid**2) / dof)
            xbar = np.mean(x)
            Sxx = np.sum((x - xbar) ** 2)

            if Sxx > 0:
                tval = t.ppf(0.975, dof)
                se_fit = s_err * np.sqrt(1.0 / n + (x_line - xbar) ** 2 / Sxx)
                ci_low = y_line - tval * se_fit
                ci_high = y_line + tval * se_fit
                ax.fill_between(x_line, ci_low, ci_high, color=color, alpha=0.18, linewidth=0, zorder=1)

        ax.plot(x_line, y_line, linestyle="--", linewidth=2.0, color=color, zorder=2)

        # annotation
        slope_decade = slope * 10.0
        slope_million = slope_decade / 1e6
        p_part = (r"$P$ < 0.05" if (np.isfinite(p_mk) and p_mk < 0.05) else rf"$P$ = {p_mk:.2f}")
        text_str = f"Trend: {slope_million:.1f} million decade$^{{-1}}$, {p_part}"
        tx, ty0, tdy = TEXT_POS.get(region, (0.02, 0.98, 0.06))

        ax.text(tx, ty0 - i * tdy, text_str, transform=ax.transAxes,
                fontsize=ANN_FONTSIZE, va="top", ha="left", color="black")

    ax.set_xlabel("Year", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Exposure to\nTC rainfall", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    apply_common_yaxis(
        ax,
        exp=common_exp,
        y_lim_scaled=common_ylim_scaled,
        y_ticks_scaled=common_yticks_scaled,
        tick_fs=TICK_FONTSIZE,
        exp_fs=EXP_FONTSIZE,
        fontname="Arial"
    )

    plt.tight_layout()
    out_fig_path = os.path.join(FIG_DIR, TS_OUTNAME[region])
    fig.savefig(out_fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
