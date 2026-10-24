#!/usr/bin/env python3
"""
Đo diện tích bề mặt đồng xu từ một ảnh (ảnh điện thoại hoặc ảnh camera Basler).

Hai cách tính diện tích:
  Cách A - Đếm pixel   : đếm số pixel bên trong viền đồng xu, nhân diện tích 1 pixel.
  Cách B - Qua đường kính: tìm biên dưới pixel theo 720 tia, loại điểm lệch,
                          khớp đường tròn -> đường kính d -> S = pi*d^2/4.

Cài đặt:  python -m pip install opencv-python numpy scipy
          (thêm pypylon nếu dùng --emu)

Cách chạy:
  1) Tìm tỉ lệ mm/pixel: chụp đồng xu cạnh một cây thước, rồi chạy
       python do_dien_tich.py dongxu.jpg --pick 10
     Mỗi điểm bấm 2 lần: bấm gần đúng trên ảnh toàn cảnh, rồi bấm chính xác
     trên ảnh phóng to x6 -> chương trình in ra mm/pixel.
  2) Đo diện tích:
       python do_dien_tich.py dongxu.jpg --mmpx 0.0425
  3) Cho ảnh đi qua camera giả lập Basler (giống khi có camera thật):
       python do_dien_tich.py dongxu.jpg --mmpx 0.0425 --emu
  Không có --mmpx thì chỉ in kết quả theo pixel.

Ảnh các bước được lưu vào thư mục ket_qua/.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares


# ----------------------------------------------------------------- nhập ảnh
def doc_anh_tu_file(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Không đọc được ảnh: {path}")
    return img


def doc_anh_qua_camera_gia_lap(path):
    """Ảnh đi qua pylon giống camera thật: pylon -> pypylon -> mảng numpy Mono8."""
    os.environ["PYLON_CAMEMU"] = "1"
    from pypylon import pylon
    src = doc_anh_tu_file(path)
    tmp = os.path.abspath("_anh_cho_emu.png")
    cv2.imwrite(tmp, src)
    with pylon.InstantCamera(pylon.FirstFound) as cam:
        cam.PixelFormat.TrySetValue("Mono8")
        cam.Width.TrySetValue(src.shape[1])
        cam.Height.TrySetValue(src.shape[0])
        cam.TestImageSelector.TrySetValue("Off")
        cam.ImageFileMode.TrySetValue("On")
        cam.ImageFilename.TrySetValue(tmp)
        with cam.GrabOne(5000) as r:
            if not r.GrabSucceeded():
                sys.exit("Camera giả lập chụp lỗi.")
            return r.Array.copy()


# ------------------------------------------------------- tìm tỉ lệ bằng thước
def _cho_bam(ten, anh):
    """Hiện ảnh trong cửa sổ, chờ 1 lần bấm chuột trái, trả về toạ độ (x, y) trên ảnh hiển thị."""
    diem = []

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            diem.append((x, y))

    cv2.namedWindow(ten, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(ten, anh)
    cv2.setMouseCallback(ten, on_click)
    while not diem:
        if cv2.waitKey(30) == 27:  # Esc để thoát
            cv2.destroyAllWindows()
            sys.exit("Đã hủy.")
    cv2.destroyWindow(ten)
    return diem[0]


def _kich_thuoc_man_hinh():
    """Trả về (rộng, cao) màn hình theo pixel thật. Trên Windows tắt phóng to DPI
    (125%, 150%) để cửa sổ OpenCV không bị phóng to vượt khỏi màn hình."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:  # không phải Windows
        return 1280, 800


def chon_hai_diem(img, do_dai_mm, zoom=6):
    """Bấm 2 điểm trên thước cách nhau do_dai_mm. Mỗi điểm bấm 2 lần:
    lần 1 bấm gần đúng trên ảnh toàn cảnh, lần 2 bấm chính xác trên ảnh phóng to."""
    h, w = img.shape
    mw, mh = _kich_thuoc_man_hinh()
    cao_toi_da, rong_toi_da = int(mh * 0.75), int(mw * 0.85)  # chừa chỗ thanh tiêu đề, taskbar
    s = min(1.0, cao_toi_da / h, rong_toi_da / w)     # thu nhỏ vừa màn hình (chỉ để nhìn)
    o_zoom = max(20, min(cao_toi_da, rong_toi_da) // (2 * zoom))  # nửa cạnh vùng phóng to
    toan = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_GRAY2BGR)
    pts = []
    for i in (1, 2):
        # Lần 1: bấm gần đúng trên ảnh toàn cảnh
        x, y = _cho_bam(f"Diem {i}/2 - bam GAN DUNG vach thuoc (Esc = huy)", toan)
        cx, cy = x / s, y / s
        # Lần 2: cắt vùng quanh điểm đó trên ẢNH GỐC, phóng to để bấm chính xác
        x0 = int(np.clip(cx - o_zoom, 0, max(0, w - 2 * o_zoom)))
        y0 = int(np.clip(cy - o_zoom, 0, max(0, h - 2 * o_zoom)))
        crop = img[y0:y0 + 2 * o_zoom, x0:x0 + 2 * o_zoom]
        to = cv2.cvtColor(cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST),
                          cv2.COLOR_GRAY2BGR)
        for k in range(0, to.shape[1], zoom * 10):     # lưới 10 pixel gốc cho dễ nhìn
            cv2.line(to, (k, 0), (k, to.shape[0]), (60, 60, 60), 1)
        for k in range(0, to.shape[0], zoom * 10):
            cv2.line(to, (0, k), (to.shape[1], k), (60, 60, 60), 1)
        zx, zy = _cho_bam(f"Diem {i}/2 - PHONG TO x{zoom}: bam CHINH XAC vao vach", to)
        px_x, px_y = x0 + (zx + 0.5) / zoom - 0.5, y0 + (zy + 0.5) / zoom - 0.5
        pts.append((px_x, px_y))
        cv2.circle(toan, (int(px_x * s), int(px_y * s)), 5, (0, 0, 255), -1)
        print(f"  Điểm {i}: x = {px_x:.1f}, y = {px_y:.1f} (pixel ảnh gốc)")

    px = math.dist(pts[0], pts[1])
    mmpx = do_dai_mm / px
    cv2.line(toan, tuple(int(v * s) for v in pts[0]), tuple(int(v * s) for v in pts[1]), (0, 0, 255), 2)
    cv2.imshow("Kiem tra 2 diem (bam phim bat ky de tiep tuc)", toan)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Khoảng cách 2 điểm: {px:.1f} pixel = {do_dai_mm} mm")
    print(f"=> Tỉ lệ: {mmpx:.5f} mm/pixel  (dùng: --mmpx {mmpx:.5f})")
    return mmpx


# -------------------------------------------------------- tách đồng xu khỏi nền
def tach_dong_xu(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Nền chiếm phần lớn viền ảnh -> nếu viền ảnh trắng thì đảo để đồng xu là màu trắng
    vien = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if vien.mean() > 127:
        bw = 255 - bw
    # Mở (opening) với nhân ~2% cạnh ảnh: cắt các chỗ dính mảnh giữa đồng xu và vật bên cạnh (thước)
    n = max(9, int(min(gray.shape) * 0.02)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = gray.shape
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.002 * h * w:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue  # chạm mép ảnh -> không phải đồng xu trọn vẹn
        per = cv2.arcLength(c, True)
        tron = 4 * math.pi * area / (per * per)  # 1.0 = tròn hoàn hảo
        if tron > best_score:
            best, best_score = c, tron
    if best is not None and best_score >= 0.6:
        (cx0, cy0), r0 = cv2.minEnclosingCircle(best)
        return blur, bw, cx0, cy0, r0, best_score

    # Dự phòng: đồng xu dính vào vật khác (ví dụ nằm sát thước) hoặc mặt đồng xu sáng gần bằng nền.
    # Tìm viền tròn bằng biến đổi Hough (dựa vào biên, không phụ thuộc màu mặt đồng xu).
    m = min(h, w)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=m * 0.05,
                               param1=60, param2=0.85,
                               minRadius=int(m * 0.03), maxRadius=int(m * 0.45))
    if circles is None:
        sys.exit("Không tìm thấy đồng xu. Chụp lại: nền trơn, đồng xu cách xa vật khác, không chạm mép ảnh.")
    cands = [c for c in circles[0]
             if c[0] - c[2] > 0 and c[1] - c[2] > 0 and c[0] + c[2] < w and c[1] + c[2] < h]
    if not cands:
        sys.exit("Đồng xu chạm mép ảnh. Chụp lại.")
    cx0, cy0, r0 = max(cands, key=lambda c: c[2])  # đường tròn lớn nhất nằm trọn trong ảnh
    print("(Đồng xu dính vào vật bên cạnh hoặc lẫn với nền - dùng biến đổi Hough để tìm)")
    return blur, bw, float(cx0), float(cy0), float(r0), float("nan")


# ------------------------------------------------ cách B: biên dưới pixel + khớp
def bien_duoi_pixel(blur, cx, cy, r0, so_tia=720, rong=0.12):
    """Dọc mỗi tia từ tâm, biên = chỗ độ sáng thay đổi mạnh nhất (đỉnh gradient),
    vị trí đỉnh được tinh chỉnh dưới pixel bằng nội suy parabol qua 3 điểm."""
    goc = np.linspace(0, 2 * np.pi, so_tia, endpoint=False)
    buoc = 0.25
    rr = np.arange(r0 * (1 - rong), r0 * (1 + rong), buoc)
    xs = (cx + np.outer(np.cos(goc), rr)).astype(np.float32)
    ys = (cy + np.outer(np.sin(goc), rr)).astype(np.float32)
    prof = cv2.remap(blur.astype(np.float32), xs, ys, cv2.INTER_LINEAR)  # nội suy song tuyến
    grad = np.abs(np.gradient(prof, buoc, axis=1))

    diem, do_sang_bien = [], []
    for i in range(so_tia):
        g = grad[i]
        j = int(np.argmax(g[1:-1])) + 1                 # đỉnh gradient mạnh nhất trên tia
        a, b, c = g[j - 1], g[j], g[j + 1]
        dmau = a - 2 * b + c
        lech = 0.5 * (a - c) / dmau if dmau != 0 else 0.0  # đỉnh parabol, trong khoảng ±0.5 bước
        r = rr[j] + lech * buoc
        diem.append((cx + r * np.cos(goc[i]), cy + r * np.sin(goc[i])))
        do_sang_bien.append(np.interp(r, rr, prof[i]))
    return np.array(diem), float(np.median(do_sang_bien))


def khop_duong_tron(pts):
    """Khớp đại số (Kasa) lấy giá trị đầu, rồi khớp hình học (least squares), loại điểm lệch 3σ."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = math.sqrt(sol[2] + cx * cx + cy * cy)

    giu = np.ones(len(pts), bool)
    for _ in range(3):
        f = lambda p: np.hypot(x[giu] - p[0], y[giu] - p[1]) - p[2]
        cx, cy, r = least_squares(f, [cx, cy, r]).x
        res = np.hypot(x - cx, y - cy) - r
        moi = np.abs(res - res[giu].mean()) < 3 * res[giu].std()
        if (moi == giu).all():
            break
        giu = moi
    return cx, cy, r, giu, res


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Đo diện tích đồng xu từ ảnh")
    ap.add_argument("image", help="ảnh đồng xu (jpg/png)")
    ap.add_argument("--mmpx", type=float, help="tỉ lệ mm/pixel")
    ap.add_argument("--pick", type=float, metavar="MM", help="bấm 2 điểm trên thước cách nhau MM để tìm mm/pixel")
    ap.add_argument("--emu", action="store_true", help="cho ảnh đi qua camera giả lập Basler (cần pypylon)")
    ap.add_argument("--out", default="ket_qua", help="thư mục lưu ảnh các bước")
    args = ap.parse_args()

    gray = doc_anh_qua_camera_gia_lap(args.image) if args.emu else doc_anh_tu_file(args.image)
    print(f"Ảnh: {gray.shape[1]} x {gray.shape[0]} pixel")

    if args.pick:
        args.mmpx = chon_hai_diem(gray, args.pick)

    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(os.path.join(args.out, "buoc1_anh_xam.png"), gray)

    blur, bw, cx0, cy0, r0, tron = tach_dong_xu(gray)
    cv2.imwrite(os.path.join(args.out, "buoc2_nhi_phan.png"), bw)

    # Cách B: biên dưới pixel + khớp đường tròn
    pts, nguong_bien = bien_duoi_pixel(blur, cx0, cy0, r0)
    cx, cy, r, giu, res = khop_duong_tron(pts)
    d_px = 2 * r
    S_B_px = math.pi * r * r

    # Cách A: đếm pixel, ngưỡng đặt bằng độ sáng đo được ngay tại biên
    vong_trong = np.zeros_like(gray)
    cv2.circle(vong_trong, (round(cx), round(cy)), round(r * 0.9), 255, -1)
    kieu = cv2.THRESH_BINARY_INV if blur[vong_trong > 0].mean() < nguong_bien else cv2.THRESH_BINARY
    _, bw2 = cv2.threshold(blur, nguong_bien, 255, kieu)
    vung = np.zeros_like(bw2)  # chỉ xét quanh đồng xu, tránh dính vật bên cạnh
    cv2.circle(vung, (round(cx), round(cy)), round(r * 1.06), 255, -1)
    bw2 = cv2.bitwise_and(bw2, vung)
    cnts, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    chua_tam = [c for c in cnts if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0]
    contour = max(chua_tam or cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    cv2.imwrite(os.path.join(args.out, "buoc3_mat_na_dong_xu.png"), mask)
    S_A_px = int(cv2.countNonZero(mask))

    print(f"Độ tròn của viền: {tron:.3f} (1 = tròn hoàn hảo)")
    print(f"Điểm biên dưới pixel: {len(pts)} điểm, loại {int((~giu).sum())} điểm lệch, "
          f"độ lệch còn lại {res[giu].std():.3f} pixel")
    print()
    print(f"Cách A - Đếm pixel     : S = {S_A_px:,.0f} pixel²")
    print(f"Cách B - Qua đường kính: d = {d_px:.2f} pixel, S = {S_B_px:,.0f} pixel²")
    print(f"Chênh lệch A so với B  : {(S_A_px - S_B_px) / S_B_px * 100:+.2f} %")
    if args.mmpx:
        k = args.mmpx
        print()
        print(f"Với tỉ lệ {k:.5f} mm/pixel:")
        print(f"  Cách A: S = {S_A_px * k * k:.2f} mm²  (đường kính tương đương {2 * math.sqrt(S_A_px / math.pi) * k:.3f} mm)")
        print(f"  Cách B: S = {S_B_px * k * k:.2f} mm²  (đường kính {d_px * k:.3f} mm)")

    # Ảnh kết quả
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = max(2, gray.shape[1] // 800)
    cv2.drawContours(vis, [contour], -1, (0, 200, 0), t)                       # xanh lá: viền cách A
    cv2.circle(vis, (round(cx), round(cy)), round(r), (0, 0, 255), t)          # đỏ: đường tròn cách B
    for (px, py) in pts[~giu]:
        cv2.circle(vis, (round(px), round(py)), 3 * t, (255, 0, 255), -1)      # tím: điểm bị loại
    cv2.drawMarker(vis, (round(cx), round(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20 * t, t)
    cv2.imwrite(os.path.join(args.out, "buoc4_ket_qua.png"), vis)
    print(f"\nĐã lưu ảnh các bước vào thư mục '{args.out}/'")


if __name__ == "__main__":
    main()
