import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# ====================== 1. Experiment Configuration ======================
np.random.seed(42)
n_points = 100

# True value = standard value at 25℃ (reference value)
true_ec_25 = np.linspace(100, 120, n_points)
# Temperature: rises linearly from 25℃ to 35℃
true_temp = np.linspace(25, 35, n_points)
raw_temp = true_temp + np.random.normal(0, 0.5, n_points)

# Generate noisy measurement: true_value * temperature_coefficient + noise
temp_factor = 1.0 + 0.0185 * (true_temp - 25.0)
raw_ec = true_ec_25 * temp_factor + np.random.normal(0, 8.0, n_points)

# Other sensors
true_ph = np.linspace(7.0, 7.2, n_points)
raw_ph = true_ph + np.random.normal(0, 0.4, n_points)
true_turb = np.linspace(500, 550, n_points)
raw_turb = true_turb + np.random.normal(0, 50.0, n_points)


# ====================== 2. EMA + Kalman Fusion Algorithm ======================
def ema_kalman_fusion(data, ema_w=0.25, q=0.02, r=8.0):
    out = np.zeros_like(data)
    ema_prev = data[0]
    kx_prev = data[0]
    kp_prev = 1.0
    for i in range(len(data)):
        # EMA (Exponential Moving Average)
        ema = ema_prev * (1 - ema_w) + data[i] * ema_w
        # 1D Kalman Filter
        pred_x = kx_prev
        pred_p = kp_prev + q
        k_gain = pred_p / (pred_p + r)
        kx = pred_x + k_gain * (ema - pred_x)
        kp = (1 - k_gain) * pred_p

        out[i] = kx
        ema_prev = ema
        kx_prev = kx
        kp_prev = kp
    return out


# ====================== 3. EC Temperature Compensation ======================
def ec_temp_compensate(ec_measured, temp_val):
    return ec_measured / (1.0 + 0.0185 * (temp_val - 25.0))


# ====================== 4. Run Algorithms ======================
ph_fused = ema_kalman_fusion(raw_ph, r=6.0)
ec_fused = ema_kalman_fusion(raw_ec, r=8.0)
ec_final = ec_temp_compensate(ec_fused, raw_temp)
turb_fused = ema_kalman_fusion(raw_turb, r=10.0)


# ====================== 5. RMSE Calculation ======================
def calc_rmse(est, true):
    return np.sqrt(np.mean((est - true) ** 2))


# EC Errors
err_raw_ec = calc_rmse(raw_ec, true_ec_25)
err_fusion_ec = calc_rmse(ec_fused, true_ec_25)
err_final_ec = calc_rmse(ec_final, true_ec_25)

# pH Errors
err_raw_ph = calc_rmse(raw_ph, true_ph)
err_fusion_ph = calc_rmse(ph_fused, true_ph)

# Turbidity Errors
err_raw_turb = calc_rmse(raw_turb, true_turb)
err_fusion_turb = calc_rmse(turb_fused, true_turb)

# ====================== 6. Print Error Table (English) ======================
print("=" * 65)
print("📊 Water Quality Sensor Data Processing Error Table (RMSE)")
print("=" * 65)
print(f"{'Sensor':<8} | {'Method':<20} | {'RMSE':<10} | {'Reduction Rate':<12}")
print("-" * 65)
print(f"{'pH':<8} | {'Raw Data':<20} | {err_raw_ph:.2f} | {'-'}")
print(
    f"{'pH':<8} | {'EMA+Kalman Fusion':<20} | {err_fusion_ph:.2f} | {((err_raw_ph - err_fusion_ph) / err_raw_ph) * 100:.1f}%")
print("-" * 65)
print(f"{'EC':<8} | {'Raw Data':<20} | {err_raw_ec:.2f} | {'-'}")
print(
    f"{'EC':<8} | {'Fusion Filtering':<20} | {err_fusion_ec:.2f} | {((err_raw_ec - err_fusion_ec) / err_raw_ec) * 100:.1f}%")
print(
    f"{'EC':<8} | {'Fusion+Temp Compensation':<20} | {err_final_ec:.2f} | {((err_raw_ec - err_final_ec) / err_raw_ec) * 100:.1f}%")
print("-" * 65)
print(f"{'Turbidity':<8} | {'Raw Data':<20} | {err_raw_turb:.2f} | {'-'}")
print(
    f"{'Turbidity':<8} | {'EMA+Kalman Fusion':<20} | {err_fusion_turb:.2f} | {((err_raw_turb - err_fusion_turb) / err_raw_turb) * 100:.1f}%")
print("=" * 65)

# ====================== 7. Plot Figures (100% No Overlap, Poster Quality) ======================
# 全局高清样式（彻底解决堆叠、遮挡问题）
plt.rcParams.update({
    'font.size': 14,
    'axes.linewidth': 2,
    'grid.linewidth': 1.2,
    'figure.dpi': 150,
    'figure.autolayout': False,
    'legend.framealpha': 1,
    'legend.borderpad': 0.8,
    'legend.labelspacing': 0.8
})
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 严格 2.5:1 画布比例（海报专用，绝对不堆叠）
fig_width = 20
fig_height = fig_width / 2.5  # 8，严格2.5:1
fig = plt.figure(figsize=(fig_width, fig_height))

# -------------------------- pH 子图 --------------------------
ax1 = plt.subplot(3, 1, 1)
line1_true, = ax1.plot(true_ph, 'k-', linewidth=3, label='True Value')
line1_raw, = ax1.plot(raw_ph, 'r.', markersize=6, alpha=0.8, label='Raw Data')
line1_fusion, = ax1.plot(ph_fused, '#0052CC', linewidth=2.5, label='EMA+Kalman Fusion')
ax1.set_title('pH Sensor: Raw Data vs Algorithm Processing', fontsize=16, pad=15)
ax1.set_ylabel('pH Value', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', width=1.5, labelsize=14)
# 图例外置到右侧，不遮挡曲线
ax1.legend(handles=[line1_true, line1_raw, line1_fusion],
           loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=12)

# -------------------------- EC 子图 --------------------------
ax2 = plt.subplot(3, 1, 2)
line2_true, = ax2.plot(true_ec_25, 'k-', linewidth=3, label='True Value at 25℃ (Reference)')
line2_raw, = ax2.plot(raw_ec, 'r.', markersize=6, alpha=0.8, label='Raw Measurement (Noisy + High)')
line2_fusion, = ax2.plot(ec_fused, '#0052CC', linewidth=2.5, label='Fusion Filtering (Only Denoising)')
line2_comp, = ax2.plot(ec_final, '#00A86B', linewidth=2.5, label='Fusion + Temperature Compensation')
ax2.set_title('EC Sensor: Raw / Filtering / Temperature Compensation', fontsize=16, pad=15)
ax2.set_ylabel('EC (μS/cm)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', width=1.5, labelsize=14)
# 图例外置到右侧
ax2.legend(handles=[line2_true, line2_raw, line2_fusion, line2_comp],
           loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=12)

# -------------------------- Turbidity 子图 --------------------------
ax3 = plt.subplot(3, 1, 3)
line3_true, = ax3.plot(true_turb, 'k-', linewidth=3, label='True Value')
line3_raw, = ax3.plot(raw_turb, 'r.', markersize=6, alpha=0.8, label='Raw Data')
line3_fusion, = ax3.plot(turb_fused, '#0052CC', linewidth=2.5, label='EMA+Kalman Fusion')
ax3.set_title('Turbidity Sensor: Raw Data vs Algorithm Processing', fontsize=16, pad=15)
ax3.set_ylabel('Turbidity RAW Value', fontsize=14)
ax3.set_xlabel('Sample Point', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', width=1.5, labelsize=14)
# 图例外置到右侧
ax3.legend(handles=[line3_true, line3_raw, line3_fusion],
           loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=12)

# 调整布局：给右侧图例留出空间，子图垂直间距充足，绝对不堆叠
plt.subplots_adjust(
    left=0.08,    # 左侧留白
    right=0.82,   # 右侧给图例留空间
    top=0.92,     # 顶部留白
    bottom=0.1,   # 底部留白
    hspace=0.45   # 子图垂直间距，彻底解决标题堆叠
)

plt.show()