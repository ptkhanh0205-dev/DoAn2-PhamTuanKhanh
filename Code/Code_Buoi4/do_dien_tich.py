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
def doc_anh_tu_file(path, kenh="xam"):
    """kenh = "xam": ảnh xám (độ sáng) - dùng cho đồng xu kim loại.
    kenh = "bao_hoa": kênh S (độ bão hòa màu, hệ HSV) - dùng cho vật CÓ MÀU trên nền trắng/xám:
    nền và bóng đổ gần như không màu (S thấp), vật có màu đậm (S cao)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Không đọc được ảnh: {path}")
    if kenh == "bao_hoa":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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
NGUONG_TRON = 0.8
NGUONG_LAP_DAY = 0.93


def tach_dong_xu(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Nền chiếm phần lớn viền ảnh -> nếu viền ảnh trắng thì đảo để đồng xu là màu trắng
    vien = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if vien.mean() > 127:
        bw = 255 - bw
    # Không dùng phép hình thái học: hệ chụp cố định, chỉ có đồng xu trên nền sạch.

    h, w = gray.shape
    # Hai lớp kiểm tra để chỉ nhận vật hình tròn:
    #   1) độ tròn C = 4π·S/P² >= NGUONG_TRON  (hình tròn trên ảnh pixel ~0,88-0,90; hình vuông ~0,79)
    #   2) độ lấp đầy E = S / (π·r0²) >= NGUONG_LAP_DAY, với r0 là bán kính đường tròn nhỏ nhất bao vùng
    #      (hình tròn ~0,96-1,0; bát giác ~0,90; elip dẹt 10% ~0,90; hình vuông ~0,64)
    # Xét cả ảnh nhị phân và ảnh đảo: khi ảnh có 3 vùng sáng khác nhau (bàn, tờ giấy, đồng xu),
    # Otsu có thể xếp đồng xu chung lớp với mặt bàn -> đồng xu thành "lỗ đen" trên tờ giấy trắng.
    best, best_score, best_ld, ung_vien, best_bw = None, 0.0, 0.0, 0, bw
    for anh in (bw, 255 - bw):
        # RETR_CCOMP: lấy cả vật nằm lọt trong lỗ của vật khác (đồng xu trên tờ giấy, giấy trên bàn);
        # chỉ giữ đường bao ngoài (hier[i][3] == -1), bỏ đường bao lỗ.
        contours, hier = cv2.findContours(anh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is None:
            continue
        for c, hh in zip(contours, hier[0]):
            if hh[3] != -1:
                continue
            area = cv2.contourArea(c)
            if area < 0.002 * h * w:
                continue  # quá nhỏ: bụi, vết bẩn
            x, y, cw, ch = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
                continue  # chạm mép ảnh -> không phải vật trọn vẹn
            ung_vien += 1
            per = cv2.arcLength(c, True)
            tron = 4 * math.pi * area / (per * per)
            (_, _), rb = cv2.minEnclosingCircle(c)
            lap_day = area / (math.pi * rb * rb)
            if tron >= NGUONG_TRON and lap_day >= NGUONG_LAP_DAY and tron > best_score:
                best, best_score, best_ld, best_bw = c, tron, lap_day, anh
    bw = best_bw
    if best is not None:
        (cx0, cy0), r0 = cv2.minEnclosingCircle(best)
        print(f"Độ lấp đầy đường tròn bao: {best_ld:.3f} (1 = tròn hoàn hảo)")
        return blur, bw, cx0, cy0, r0, best_score

    # Không có vật nào đạt cả hai điều kiện: dừng lại thay vì đoán.
    if ung_vien == 0:
        sys.exit("Không tìm thấy đồng xu trong ảnh. Kiểm tra: có đặt đồng xu chưa, nền trơn, "
                 "đồng xu không chạm mép ảnh.")
    sys.exit(f"Không tìm thấy đồng xu: có {ung_vien} vật trong ảnh nhưng không vật nào là hình tròn "
             f"(cần độ tròn >= {NGUONG_TRON} và độ lấp đầy >= {NGUONG_LAP_DAY}). "
             "Kiểm tra: đồng xu có dính vật khác hoặc bị bóng che không.")


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
    ap.add_argument("--he-chup", metavar="FILE",
                    help="file hệ chụp cố định (he_chup.npz) từ hieu_chuan_ban_co.py co_dinh: nắn ảnh và lấy tỉ lệ")
    ap.add_argument("--kenh", choices=["xam", "bao_hoa"], default="xam",
                    help="xam: ảnh xám (mặc định, đồng xu); bao_hoa: độ bão hòa màu (vật có màu, vd nắp chai)")
    ap.add_argument("--out", default="ket_qua", help="thư mục lưu ảnh các bước")
    args = ap.parse_args()

    gray = doc_anh_qua_camera_gia_lap(args.image) if args.emu else doc_anh_tu_file(args.image, args.kenh)
    print(f"Ảnh: {gray.shape[1]} x {gray.shape[0]} pixel, kênh: "
          f"{'độ bão hòa màu (S)' if args.kenh == 'bao_hoa' else 'ảnh xám'}")

    if args.he_chup:
        hc = np.load(args.he_chup)
        if tuple(hc["anh_size"]) != (gray.shape[1], gray.shape[0]):
            sys.exit(f"Ảnh {gray.shape[1]} x {gray.shape[0]} khác độ phân giải lúc hiệu chuẩn "
                     f"{tuple(hc['anh_size'])}. Chụp lại đúng chế độ, hoặc hiệu chuẩn lại.")
        gray = cv2.warpPerspective(gray, hc["M"], tuple(int(v) for v in hc["out_size"]),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        args.mmpx = float(hc["mmpx"])
        print(f"Đã nắn ảnh theo {args.he_chup}: {gray.shape[1]} x {gray.shape[0]} pixel, "
              f"tỉ lệ {args.mmpx:.6f} mm/pixel")

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
