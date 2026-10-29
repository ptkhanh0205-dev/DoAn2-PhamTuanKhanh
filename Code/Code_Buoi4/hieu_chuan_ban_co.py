#!/usr/bin/env python3
"""
Hiệu chuẩn bàn cờ cho việc đo đồng xu bằng ảnh (điện thoại hoặc camera Basler).

Ba lệnh:

  1) Tạo bàn cờ để in (in đúng tỉ lệ 100%, KHÔNG chọn "Fit to page"):
       python hieu_chuan_ban_co.py tao --o 5 --cot 10 --hang 7
     -> ban_co_10x7_o5mm.png (300 dpi). In xong đo lại cạnh ô bằng thước kẹp.

  2) Làm phẳng ảnh bằng bàn cờ đặt CẠNH đồng xu trong cùng 1 ảnh (cách nên dùng với điện thoại):
       python hieu_chuan_ban_co.py phang anh.jpg --o 5 --goc 9x6
     Chương trình tìm 54 góc bàn cờ, tính phép biến đổi phối cảnh (homography) từ ảnh về
     mặt phẳng thật, rồi "nắn" ảnh thành ảnh nhìn thẳng từ trên xuống với tỉ lệ cố định.
     -> anh_phang.png và in ra lệnh đo:  python do_dien_tich.py anh_phang.png --mmpx 0.05
     Cách này sửa được cả việc chụp nghiêng (tỉ lệ đều trên toàn ảnh).

  3) Hệ chụp cố định (camera/điện thoại gắn trên giá, không xê dịch):
       python hieu_chuan_ban_co.py co_dinh anh_ban_co.jpg --o 5 --goc 9x6
     Chụp bàn cờ MỘT lần, đặt đúng chỗ sẽ đặt đồng xu (kê cao bằng độ dày đồng xu).
     -> he_chup.npz. Các lần sau chỉ chụp đồng xu và chạy:
       python do_dien_tich.py anh_dong_xu.jpg --he-chup he_chup.npz

  4) Hiệu chuẩn ống kính đầy đủ theo Zhang (15-20 ảnh bàn cờ ở nhiều góc):
       python hieu_chuan_ban_co.py ong_kinh thu_muc_anh --o 5 --goc 9x6
     -> calib.npz (ma trận camera K, hệ số méo). Lệnh "phang" sẽ tự dùng file này
        để sửa méo trước nếu có (--calib calib.npz).

--goc là số GÓC TRONG (bàn cờ 10 x 7 ô có 9 x 6 góc trong).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

TIEU_CHUAN = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def doc_anh(path, mau=False):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                       cv2.IMREAD_COLOR if mau else cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Không đọc được ảnh: {path}")
    return img


def ghi_anh(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)


def doc_kich_thuoc(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def tim_goc(gray, goc):
    """Tìm góc trong của bàn cờ, chính xác dưới pixel. Trả về mảng (N, 2) hoặc None."""
    # Ảnh điện thoại rất lớn: tìm trên ảnh thu nhỏ cho nhanh, rồi tinh chỉnh trên ảnh gốc.
    s = min(1.0, 1600 / max(gray.shape))
    nho = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    ok, pts = cv2.findChessboardCornersSB(nho, goc, flags)
    if not ok:
        ok, pts = cv2.findChessboardCorners(nho, goc, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    pts = (pts.reshape(-1, 2) / s).astype(np.float32)
    win = max(3, int(round(4 / s)))
    pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (win, win), (-1, -1), TIEU_CHUAN)
    return pts.reshape(-1, 2)


def diem_that(goc, o_mm):
    """Toạ độ thật (mm) của các góc, theo đúng thứ tự OpenCV trả về (theo hàng)."""
    c, h = goc
    xs, ys = np.meshgrid(np.arange(c), np.arange(h))
    return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32) * o_mm


# ------------------------------------------------------------------ lệnh "tao"
def lenh_tao(a):
    dpi = 300
    px_mm = dpi / 25.4
    o = int(round(a.o * px_mm))
    w, h = a.cot * o, a.hang * o
    le = int(round(10 * px_mm))  # lề trắng 10 mm
    img = np.full((h + 2 * le, w + 2 * le), 255, np.uint8)
    for r in range(a.hang):
        for c in range(a.cot):
            if (r + c) % 2 == 0:
                img[le + r * o: le + (r + 1) * o, le + c * o: le + (c + 1) * o] = 0
    cv2.putText(img, f"{a.cot}x{a.hang} o, o = {a.o} mm - in 100%, khong Fit to page",
                (le, le + h + int(6 * px_mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    ten = a.out or f"ban_co_{a.cot}x{a.hang}_o{a.o:g}mm.png"
    # Ghi kèm dpi để máy in in đúng kích thước
    try:
        from PIL import Image
        Image.fromarray(img).save(ten, dpi=(dpi, dpi))
    except ImportError:
        ghi_anh(ten, img)
    print(f"Đã tạo {ten}: {a.cot} x {a.hang} ô ({a.cot - 1} x {a.hang - 1} góc trong), ô {a.o} mm, {dpi} dpi.")
    print("In đúng tỉ lệ 100%, rồi đo lại cạnh ô bằng thước kẹp và dùng số đo thật cho --o.")


# ------------------------------------------------------------ lệnh "ong_kinh"
def lenh_ong_kinh(a):
    goc = doc_kich_thuoc(a.goc)
    files = sorted(sum([glob.glob(os.path.join(a.thu_muc, e)) for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")], []))
    if not files:
        sys.exit("Không có ảnh trong thư mục.")
    P3, P2, size = [], [], None
    obj = np.hstack([diem_that(goc, a.o), np.zeros((goc[0] * goc[1], 1), np.float32)])
    for f in files:
        g = doc_anh(f)
        if size is None:
            size = g.shape[::-1]
        elif g.shape[::-1] != size:
            print(f"  Bỏ {os.path.basename(f)}: khác độ phân giải")
            continue
        pts = tim_goc(g, goc)
        print(f"  {os.path.basename(f)}: {'tìm được góc' if pts is not None else 'KHÔNG tìm được góc'}")
        if pts is not None:
            P3.append(obj)
            P2.append(pts.reshape(-1, 1, 2))
    if len(P2) < 5:
        sys.exit("Cần ít nhất 5 ảnh tìm được góc (nên 15-20).")
    rms, K, dist, _, _ = cv2.calibrateCamera(P3, P2, size, None, None)
    np.savez(a.out, K=K, dist=dist, size=np.array(size))
    print(f"\nDùng {len(P2)} ảnh. Sai số chiếu lại RMS = {rms:.3f} pixel (dưới 0,5 là tốt).")
    print(f"fx = {K[0,0]:.1f}, fy = {K[1,1]:.1f}, cx = {K[0,2]:.1f}, cy = {K[1,2]:.1f}")
    print("Hệ số méo k1, k2, p1, p2, k3 =", np.round(dist.ravel(), 5))
    print(f"Đã lưu {a.out}")


# --------------------------------------------------------------- lệnh "phang"
def lenh_phang(a):
    goc = doc_kich_thuoc(a.goc)
    img = doc_anh(a.anh, mau=True)
    if a.calib and os.path.exists(a.calib):
        d = np.load(a.calib)
        if tuple(d["size"]) == img.shape[1::-1]:
            img = cv2.undistort(img, d["K"], d["dist"])
            print(f"Đã sửa méo ống kính bằng {a.calib}")
        else:
            print(f"Bỏ qua {a.calib}: khác độ phân giải ảnh")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = tim_goc(gray, goc)
    if pts is None:
        sys.exit("Không tìm được bàn cờ. Kiểm tra --goc (số góc TRONG), bàn cờ phẳng, đủ sáng, không bị che.")
    that = diem_that(goc, a.o)

    # Tỉ lệ thô ngay trên ảnh gốc (để so sánh): khoảng cách góc kề nhau
    lu = pts.reshape(goc[1], goc[0], 2)
    d_ngang = np.linalg.norm(np.diff(lu, axis=1), axis=2).ravel()
    d_doc = np.linalg.norm(np.diff(lu, axis=0), axis=2).ravel()
    dd = np.concatenate([d_ngang, d_doc])
    print(f"Tìm được {len(pts)} góc. Trên ảnh gốc: 1 ô = {dd.mean():.2f} px "
          f"(nhỏ nhất {dd.min():.2f}, lớn nhất {dd.max():.2f}; chênh {100*(dd.max()-dd.min())/dd.mean():.1f}% do nghiêng/méo)")

    # Homography: ảnh -> mặt phẳng thật (mm) -> ảnh phẳng với ppmm pixel/mm
    ppmm = float(a.ppmm) if a.ppmm else float(round(float(dd.mean()) / a.o, 1))  # giữ độ phân giải gần như ảnh gốc
    H, _ = cv2.findHomography(pts, that * ppmm, 0)
    # Đặt khung ảnh đầu ra bao trọn toàn bộ ảnh gốc
    h, w = gray.shape
    goc4 = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
    x0, y0 = np.floor(goc4.min(0))
    x1, y1 = np.ceil(goc4.max(0))
    if (x1 - x0) * (y1 - y0) > 60e6:
        sys.exit("Ảnh phẳng quá lớn (camera quá nghiêng?). Thử chụp thẳng hơn hoặc giảm --ppmm.")
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    phang = cv2.warpPerspective(img, T @ H, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_CUBIC,
                                borderValue=(255, 255, 255))

    # Kiểm tra: chiếu góc sang ảnh phẳng, so với lưới thật
    p_phang = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), T @ H).reshape(-1, 2)
    ly_tuong = that * ppmm - [x0, y0]
    loi = np.linalg.norm(p_phang - ly_tuong, axis=1)
    mmpx = 1.0 / ppmm

    # Che bàn cờ (tô trắng) để chương trình đo không nhận nhầm bàn cờ là vật thể
    if not a.giu_ban_co:
        c4 = np.float32([[-1, -1], [goc[0], -1], [goc[0], goc[1]], [-1, goc[1]]]) * a.o * ppmm - [x0, y0]
        tam = c4.mean(0)
        c4 = tam + (c4 - tam) * 1.08  # nới rộng thêm để che cả viền ô ngoài
        cv2.fillPoly(phang, [c4.astype(np.int32)], (255, 255, 255))

    ghi_anh(a.out, phang)
    print(f"Ảnh phẳng: {phang.shape[1]} x {phang.shape[0]} px, tỉ lệ {ppmm} px/mm "
          f"= {mmpx:.6f} mm/pixel (đều trên toàn ảnh, tại mặt phẳng bàn cờ)")
    print(f"Sai lệch góc bàn cờ sau khi nắn: trung bình {loi.mean():.2f} px, lớn nhất {loi.max():.2f} px")
    print(f"Đã lưu {a.out}\n\nĐo diện tích:\n  python do_dien_tich.py {a.out} --mmpx {mmpx:.6f}")


# ------------------------------------------------------------ lệnh "co_dinh"
def lenh_co_dinh(a):
    """Hệ chụp cố định: chụp bàn cờ MỘT lần, lưu phép nắn ảnh + tỉ lệ vào file.
    Các lần sau chỉ chụp đồng xu, do_dien_tich.py đọc file này để nắn ảnh và đổi sang mm."""
    goc = doc_kich_thuoc(a.goc)
    img = doc_anh(a.anh, mau=True)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = tim_goc(gray, goc)
    if pts is None:
        sys.exit("Không tìm được bàn cờ. Kiểm tra --goc (số góc TRONG), bàn cờ phẳng, đủ sáng, không bị che.")
    that = diem_that(goc, a.o)
    lu = pts.reshape(goc[1], goc[0], 2)
    dd = np.concatenate([np.linalg.norm(np.diff(lu, axis=1), axis=2).ravel(),
                         np.linalg.norm(np.diff(lu, axis=0), axis=2).ravel()])
    chenh = 100 * (dd.max() - dd.min()) / dd.mean()
    ppmm = float(a.ppmm) if a.ppmm else float(round(float(dd.mean()) / a.o, 1))
    H, _ = cv2.findHomography(pts, that * ppmm, 0)
    h, w = gray.shape
    goc4 = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
    x0, y0 = np.floor(goc4.min(0))
    x1, y1 = np.ceil(goc4.max(0))
    if (x1 - x0) * (y1 - y0) > 60e6:
        sys.exit("Ảnh nắn quá lớn (camera quá nghiêng?). Chỉnh camera thẳng hơn hoặc giảm --ppmm.")
    M = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ H
    p_phang = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
    loi = np.linalg.norm(p_phang - (that * ppmm - [x0, y0]), axis=1)
    np.savez(a.out, M=M, out_size=np.array([int(x1 - x0), int(y1 - y0)]),
             anh_size=np.array([w, h]), mmpx=1.0 / ppmm, o_mm=a.o, goc=np.array(goc))
    print(f"Tìm được {len(pts)} góc. 1 ô = {dd.mean():.2f} px; chênh giữa các ô {chenh:.1f}%")
    if chenh > 3:
        print("  Lưu ý: các ô chênh nhau nhiều -> camera đang nghiêng. Chương trình vẫn nắn được, "
              "nhưng nên chỉnh camera thẳng hơn.")
    print(f"Sai lệch góc sau khi nắn: trung bình {loi.mean():.2f} px, lớn nhất {loi.max():.2f} px")
    print(f"Tỉ lệ sau khi nắn: {1.0 / ppmm:.6f} mm/pixel")
    print(f"Đã lưu {a.out}. Từ giờ, KHÔNG xê dịch camera; mỗi lần đo chỉ cần:")
    print(f"  python do_dien_tich.py anh_dong_xu.jpg --he-chup {a.out}")


def lenh_so_sanh(a):
    """So 2 ảnh bàn cờ (đầu buổi và cuối buổi) để biết camera có bị xê dịch không.
    Bàn cờ đặt lại khó trùng đúng chỗ cũ, nên không so vị trí góc mà so KÍCH THƯỚC Ô (pixel/ô):
    camera bị nâng/hạ hoặc nghiêng thì kích thước ô thay đổi -> tỉ lệ mm/pixel không còn đúng."""
    goc = doc_kich_thuoc(a.goc)
    kq = []
    for f in (a.anh1, a.anh2):
        g = cv2.cvtColor(doc_anh(f, mau=True), cv2.COLOR_BGR2GRAY)
        p = tim_goc(g, goc)
        if p is None:
            sys.exit(f"Không tìm được bàn cờ trong {f}.")
        lu = p.reshape(goc[1], goc[0], 2)
        ngang = np.linalg.norm(np.diff(lu, axis=1), axis=2).mean()
        doc = np.linalg.norm(np.diff(lu, axis=0), axis=2).mean()
        kq.append((g.shape, ngang, doc))
        print(f"{f}: 1 ô = {ngang:.2f} px (ngang), {doc:.2f} px (dọc)")
    if kq[0][0] != kq[1][0]:
        sys.exit("Hai ảnh khác kích thước -> không cùng cài đặt camera.")
    dn = 100 * abs(kq[1][1] - kq[0][1]) / kq[0][1]
    dd = 100 * abs(kq[1][2] - kq[0][2]) / kq[0][2]
    print(f"Chênh kích thước ô: ngang {dn:.2f}%, dọc {dd:.2f}%")
    if max(dn, dd) <= 0.3:
        print("=> Tỉ lệ không đổi: camera không xê dịch đáng kể. Các ảnh đồng xu dùng được.")
    else:
        print("=> Tỉ lệ đã đổi (camera bị nâng/hạ/nghiêng, hoặc bàn cờ đặt chỗ khác xa). "
              "Kết quả đo có thể lệch khoảng gấp đôi số % này về diện tích.")

def main():
    ap = argparse.ArgumentParser(description="Hiệu chuẩn bàn cờ")
    sub = ap.add_subparsers(dest="lenh", required=True)

    t = sub.add_parser("tao", help="tạo ảnh bàn cờ để in")
    t.add_argument("--o", type=float, default=5, help="cạnh ô, mm")
    t.add_argument("--cot", type=int, default=10, help="số ô theo ngang")
    t.add_argument("--hang", type=int, default=7, help="số ô theo dọc")
    t.add_argument("--out")

    k = sub.add_parser("ong_kinh", help="hiệu chuẩn ống kính từ nhiều ảnh (Zhang)")
    k.add_argument("thu_muc")
    k.add_argument("--o", type=float, required=True, help="cạnh ô thật, mm (đo bằng thước kẹp)")
    k.add_argument("--goc", default="9x6", help="số góc trong, dạng cột x hàng")
    k.add_argument("--out", default="calib.npz")

    p = sub.add_parser("phang", help="nắn ảnh có bàn cờ thành ảnh nhìn thẳng, tỉ lệ đều")
    p.add_argument("anh")
    p.add_argument("--o", type=float, required=True, help="cạnh ô thật, mm (đo bằng thước kẹp)")
    p.add_argument("--goc", default="9x6", help="số góc trong, dạng cột x hàng")
    p.add_argument("--ppmm", type=float, help="pixel/mm của ảnh phẳng (mặc định: gần bằng ảnh gốc)")
    p.add_argument("--calib", default="calib.npz", help="file hiệu chuẩn ống kính (nếu có)")
    p.add_argument("--giu-ban-co", action="store_true", help="không che bàn cờ trên ảnh phẳng")
    p.add_argument("--out", default="anh_phang.png")

    c = sub.add_parser("co_dinh", help="hệ chụp cố định: chụp bàn cờ 1 lần, lưu phép nắn và tỉ lệ")
    c.add_argument("anh", help="ảnh bàn cờ đặt đúng chỗ sẽ đặt đồng xu")
    c.add_argument("--o", type=float, required=True, help="cạnh ô thật, mm (đo bằng thước kẹp)")
    c.add_argument("--goc", default="9x6", help="số góc trong, dạng cột x hàng")
    c.add_argument("--ppmm", type=float, help="pixel/mm của ảnh nắn (mặc định: gần bằng ảnh gốc)")
    c.add_argument("--out", default="he_chup.npz")

    ss = sub.add_parser("so_sanh", help="so 2 ảnh bàn cờ xem camera có bị xê dịch không")
    ss.add_argument("anh1")
    ss.add_argument("anh2")
    ss.add_argument("--goc", default="9x6", help="số góc trong, dạng cột x hàng")

    a = ap.parse_args()
    {"so_sanh": lenh_so_sanh, "tao": lenh_tao, "ong_kinh": lenh_ong_kinh, "phang": lenh_phang, "co_dinh": lenh_co_dinh}[a.lenh](a)


if __name__ == "__main__":
    main()
